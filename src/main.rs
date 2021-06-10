use std::{
    io::Cursor,
    mem::size_of,
    path::PathBuf,
    process, ptr,
    time::{Duration, Instant},
};

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

fn measure_fork() -> Duration {
    let (pipe_rx, pipe_tx) = pipe().expect("pipe failed");

    let fork_begin = Instant::now();
    match unsafe { fork().expect("fork failed") } {
        ForkResult::Parent { child } => {
            close(pipe_tx).expect("close failed");

            let mut serialized_fork_time = [0; size_of::<u64>()];
            read(pipe_rx, &mut serialized_fork_time).expect("read failed");
            let mut reader = Cursor::new(serialized_fork_time);
            let fork_micros = reader.read_u64::<LittleEndian>().unwrap();
            let fork_time = Duration::from_micros(fork_micros);

            close(pipe_rx).expect("close failed");

            let status = wait().expect("wait failed");
            match status {
                WaitStatus::Exited(pid, exit_code) => {
                    assert_eq!(child, pid);
                    assert_eq!(exit_code, 0);
                }
                _ => panic!("unexpected status"),
            }

            fork_time
        }
        ForkResult::Child => {
            let fork_time = fork_begin.elapsed();
            close(pipe_rx).expect("close failed");

            let mut serialized_fork_micros = Vec::with_capacity(size_of::<u64>());
            serialized_fork_micros
                .write_u64::<LittleEndian>(fork_time.as_micros() as u64)
                .unwrap();
            write(pipe_tx, &serialized_fork_micros).expect("write failed");

            process::exit(0);
        }
    }
}

const WARMUP_DURATION: Duration = Duration::from_secs(1);
const MEASURE_DURATION: Duration = Duration::from_secs(5);
const CI_95: f64 = 1.96;
const MAX_FORK_MICROS: u64 = 10000;
const PAGE_SIZE: usize = 4096;

type Stats = (usize, (f64, f64, f64));

fn benchmark_fork() -> Stats {
    let mut hist = Histogram::<u16>::new_with_max(MAX_FORK_MICROS, 3).unwrap();

    println!("warmup for: {:?}", WARMUP_DURATION);
    let mut warmup_iters = 0;
    let warmup_begin = Instant::now();
    while warmup_begin.elapsed() < WARMUP_DURATION {
        measure_fork();
        warmup_iters += 1;
    }

    let speed = warmup_iters as f64 / warmup_begin.elapsed().as_secs_f64();
    println!("speed: {:.2} iters/sec", speed);

    let me = Process::myself().expect("cannot access procfs");
    let rss = me.stat.rss;
    println!("process rss: {} pages", rss);

    println!("measuring for: {:?}", MEASURE_DURATION);
    let measure_begin = Instant::now();
    while measure_begin.elapsed() < MEASURE_DURATION {
        let fork_time = measure_fork();
        hist += fork_time.as_micros() as u64;
    }

    let me = Process::myself().expect("cannot access procfs");
    println!("rss changed by: {} pages", me.stat.rss - rss);

    let mean = hist.mean();
    let half_ci = CI_95 * (hist.stdev() / (hist.len() as f64).sqrt());

    let stats = (mean - half_ci, mean, mean + half_ci);
    println!("[{:.2} {:.2} {:.2}]", stats.0, stats.1, stats.2);

    (rss as usize, stats)
}

fn benchmark_fork_with_map(pages: usize, group_pages: usize, huge_pages: bool) -> Stats {
    println!("mapping {} pages in groups of {}", pages, group_pages);

    let group_size = group_pages * PAGE_SIZE;

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
                .expect("mmap failed")
            };

            if huge_pages {
                unsafe {
                    madvise(map, group_size, MmapAdvise::MADV_HUGEPAGE).expect("madvise failed")
                };
            }

            for idx in (0..group_size).step_by(PAGE_SIZE) {
                unsafe { *map.add(idx).cast::<u8>() = 1 };
            }

            maps.push(map);
        }
    }

    let stats = benchmark_fork();

    for map in maps {
        unsafe {
            munmap(map, group_size).expect("munmap failed");
        }
    }

    stats
}

#[derive(StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    output_file: PathBuf,
    #[structopt(long)]
    huge_pages: bool,
    #[structopt(long)]
    group_pages: Option<usize>,
}

fn main() {
    let opt = Opt::from_args();

    let mut points = Vec::new();
    for pages_count in &[0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000] {
        let group_pages = opt.group_pages.unwrap_or(*pages_count);
        let (pages, (_, mean, _)) =
            benchmark_fork_with_map(*pages_count, group_pages, opt.huge_pages);
        println!();
        points.push((pages, mean));
    }

    let mut writer = Writer::from_path(opt.output_file).expect("open output file failed");
    writer
        .write_record(&["pages", "time"])
        .expect("write failed");
    for (pages, mean_time) in points {
        writer
            .write_record(&[pages.to_string(), mean_time.to_string()])
            .expect("write failed");
    }
}
