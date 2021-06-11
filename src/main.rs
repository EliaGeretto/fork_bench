use core::slice;
use std::{
    ffi::c_void,
    io::Cursor,
    mem::size_of,
    path::PathBuf,
    process, ptr,
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use csv::Writer;
use hdrhistogram::Histogram;
use nix::{
    sys::{
        mman::{madvise, mmap, munmap, MapFlags, MmapAdvise, ProtFlags},
        wait::{wait, WaitStatus},
    },
    unistd::{close, fork, pipe, read, write, ForkResult},
};
use procfs::process::Process;
use structopt::StructOpt;

const CI_95: f64 = 1.96;
const MAX_FORK_MICROS: u64 = 10000;

const TEST_PAGE_NUMS: [usize; 6] = [0, 1000, 2000, 3000, 4000, 5000];
const WARMUP_DURATION: Duration = Duration::from_secs(1);
const MEASURE_DURATION: Duration = Duration::from_secs(5);

type Stats = (usize, (f64, f64, f64));

fn measure_fork(maps: &mut [&mut [u8]], _page_size: usize) -> anyhow::Result<Duration> {
    let (pipe_rx, pipe_tx) = pipe().context("pipe failed")?;

    let fork_begin = Instant::now();
    match unsafe { fork().context("fork failed")? } {
        ForkResult::Parent { child } => {
            close(pipe_tx).context("close failed")?;

            let mut serialized_fork_time = [0; size_of::<u64>()];
            read(pipe_rx, &mut serialized_fork_time).context("read failed")?;
            let mut reader = Cursor::new(serialized_fork_time);
            let fork_micros = reader.read_u64::<LittleEndian>().unwrap();
            let fork_time = Duration::from_micros(fork_micros);

            close(pipe_rx).context("close failed")?;

            let status = wait().context("wait failed")?;
            match status {
                WaitStatus::Exited(pid, exit_code) => {
                    assert_eq!(child, pid);
                    assert_eq!(exit_code, 0);
                }
                _ => bail!("unexpected status"),
            }

            Ok(fork_time)
        }
        ForkResult::Child => {
            // In the child the program should crash if something goes wrong.
            let fork_time = fork_begin.elapsed();
            close(pipe_rx).expect("close failed");

            let mut serialized_fork_micros = Vec::with_capacity(size_of::<u64>());
            serialized_fork_micros
                .write_u64::<LittleEndian>(fork_time.as_micros() as u64)
                .unwrap();
            write(pipe_tx, &serialized_fork_micros).expect("write failed");

            // Touch all the bytes in the maps to mess with CPU caches and TLB.
            for map in maps {
                for entry in map.iter_mut() {
                    *entry = 1;
                }
            }

            process::exit(0);
        }
    }
}

fn measure_wait(_maps: &mut [&mut [u8]], _page_size: usize) -> anyhow::Result<Duration> {
    let fork_begin = Instant::now();
    match unsafe { fork().context("fork failed")? } {
        ForkResult::Parent { child } => {
            let status = wait().context("wait failed")?;
            let wait_time = fork_begin.elapsed();
            match status {
                WaitStatus::Exited(pid, exit_code) => {
                    assert_eq!(child, pid);
                    assert_eq!(exit_code, 0);
                }
                _ => bail!("unexpected status"),
            }

            Ok(wait_time)
        }
        ForkResult::Child => {
            // Touching the maps would mean corrupting the measurement
            process::exit(0);
        }
    }
}

fn benchmark_func<F>(mut measure_func: F) -> anyhow::Result<Stats>
where
    F: FnMut() -> anyhow::Result<Duration>,
{
    let mut hist =
        Histogram::<u16>::new_with_max(MAX_FORK_MICROS, 3).context("histogram creation failed")?;

    println!("warmup for: {:?}", WARMUP_DURATION);
    let mut warmup_iters = 0;
    let warmup_begin = Instant::now();
    while warmup_begin.elapsed() < WARMUP_DURATION {
        measure_func()?;
        warmup_iters += 1;
    }

    let speed = warmup_iters as f64 / warmup_begin.elapsed().as_secs_f64();
    println!("speed: {:.2} iters/sec", speed);

    let me = Process::myself().context("cannot access procfs")?;
    let rss = me.stat.rss;
    println!("process rss: {} pages", rss);

    println!("measuring for: {:?}", MEASURE_DURATION);
    let measure_begin = Instant::now();
    while measure_begin.elapsed() < MEASURE_DURATION {
        let iter_time = measure_func()?;
        hist += iter_time.as_micros() as u64;
    }

    let me = Process::myself().context("cannot access procfs")?;
    println!("rss changed by: {} pages", me.stat.rss - rss);

    let mean = hist.mean();
    let half_ci = CI_95 * (hist.stdev() / (hist.len() as f64).sqrt());

    let stats = (mean - half_ci, mean, mean + half_ci);
    println!("[{:.2} {:.2} {:.2}]", stats.0, stats.1, stats.2);

    Ok((rss as usize, stats))
}

fn benchmark_fork(maps: &mut [&mut [u8]], page_size: usize) -> anyhow::Result<Stats> {
    println!("measuring fork");
    benchmark_func(|| measure_fork(maps, page_size))
}

// This is an approximation since it includes also the time taken for the parent
// process to wake up.
fn benchmark_exit(maps: &mut [&mut [u8]], page_size: usize) -> anyhow::Result<Stats> {
    println!("measuring fork");
    let fork_stats = benchmark_func(|| measure_fork(maps, page_size))?;
    println!("measuring wait");
    let wait_stats = benchmark_func(|| measure_wait(maps, page_size))?;

    Ok((
        fork_stats.0,
        (
            wait_stats.1 .0 - fork_stats.1 .0,
            wait_stats.1 .1 - fork_stats.1 .1,
            wait_stats.1 .2 - fork_stats.1 .2,
        ),
    ))
}

fn run_benchmark_with_map<F>(
    target_function: F,
    pages: usize,
    group_pages: usize,
    page_size: usize,
    huge_pages: bool,
) -> anyhow::Result<Stats>
where
    F: FnOnce(&mut [&mut [u8]], usize) -> anyhow::Result<Stats>,
{
    let huge_pages_msg = if huge_pages { " with huge pages" } else { "" };
    println!(
        "mapping {} pages in groups of {}{}",
        pages, group_pages, huge_pages_msg
    );

    let group_size = group_pages * page_size;

    let mut maps = Vec::new();
    if pages > 0 {
        assert_eq!(pages % group_pages, 0);

        for _ in (0..pages).step_by(group_pages) {
            let map = unsafe {
                mmap(
                    ptr::null_mut(),
                    group_size,
                    ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                    MapFlags::MAP_PRIVATE | MapFlags::MAP_ANONYMOUS,
                    -1,
                    0,
                )
                .context("mmap failed")?
            };

            if huge_pages {
                unsafe {
                    madvise(map, group_size, MmapAdvise::MADV_HUGEPAGE).context("madvise failed")?
                };
            }

            let map = unsafe { slice::from_raw_parts_mut(map.cast::<u8>(), group_size) };

            for idx in (0..group_size).step_by(page_size) {
                map[idx] = 1;
            }

            maps.push(map);
        }
    }

    let stats = target_function(&mut maps, page_size);

    for map in maps {
        unsafe {
            munmap(map.as_mut_ptr().cast::<c_void>(), group_size).context("munmap failed")?;
        }
    }

    stats
}

enum Target {
    Fork,
    Exit,
}

impl FromStr for Target {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fork" => Ok(Target::Fork),
            "exit" => Ok(Target::Exit),
            _ => Err(anyhow!("unrecognized target: {}", s)),
        }
    }
}

#[derive(StructOpt)]
struct Opt {
    target: Target,
    #[structopt(parse(from_os_str))]
    output_file: PathBuf,
    #[structopt(long)]
    huge_pages: bool,
    #[structopt(long)]
    group_pages: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let target_function = match opt.target {
        Target::Fork => benchmark_fork,
        Target::Exit => benchmark_exit,
    };

    let page_size = procfs::page_size().context("retrieve page size failed")? as usize;

    let mut points = Vec::new();
    for pages_count in &TEST_PAGE_NUMS {
        let group_pages = opt.group_pages.unwrap_or(*pages_count);
        let stats = run_benchmark_with_map(
            target_function,
            *pages_count,
            group_pages,
            page_size,
            opt.huge_pages,
        )?;
        println!();
        points.push(stats);
    }

    let mut writer = Writer::from_path(opt.output_file).context("open output file failed")?;
    writer
        .write_record(&["pages", "time_lower_ci", "time_mean", "time_upper_ci"])
        .context("write failed")?;
    for (pages, time_stats) in points {
        writer
            .write_record(&[
                pages.to_string(),
                time_stats.0.to_string(),
                time_stats.1.to_string(),
                time_stats.2.to_string(),
            ])
            .context("write failed")?;
    }

    Ok(())
}
