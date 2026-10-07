use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::net::SocketAddr;
use std::path::Path;

use crate::config::OutputFormat;
use crate::features::FlowRecord;

pub struct FlowWriter {
    format: OutputFormat,
    inner: FlowWriterInner,
    backend_addr: Option<SocketAddr>,
}

enum FlowWriterInner {
    Jsonl(Box<dyn Write>),
    Csv(csv::Writer<Box<dyn Write>>),
}

impl FlowWriter {
    pub fn new(
        output_path: Option<&Path>,
        format: OutputFormat,
        backend_target: Option<&str>,
    ) -> io::Result<Self> {
        let boxed_writer: Box<dyn Write> = match output_path {
            Some(path) => {
                let file = File::create(path)?;
                Box::new(BufWriter::new(file))
            }
            None => Box::new(BufWriter::new(io::stdout())),
        };

        let inner = match format {
            OutputFormat::Jsonl => FlowWriterInner::Jsonl(boxed_writer),
            OutputFormat::Csv => FlowWriterInner::Csv(csv::Writer::from_writer(boxed_writer)),
        };

        let backend_addr = backend_target.and_then(|target| {
            let host_port = target
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .split('/')
                .next()
                .unwrap_or(target);
            match host_port.parse::<SocketAddr>() {
                Ok(addr) => Some(addr),
                Err(_) => {
                    use std::net::ToSocketAddrs;
                    host_port.to_socket_addrs().ok().and_then(|mut iter| iter.next())
                }
            }
        });

        Ok(Self {
            format,
            inner,
            backend_addr,
        })
    }

    pub fn write_record(&mut self, record: &FlowRecord) -> io::Result<()> {
        match &mut self.inner {
            FlowWriterInner::Jsonl(writer) => {
                let json_bytes = serde_json::to_vec(record)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                writer.write_all(&json_bytes)?;
                writer.write_all(b"\n")?;
                writer.flush()?;

                if let Some(addr) = self.backend_addr {
                    forward_flow_to_backend(addr, &json_bytes);
                }
            }
            FlowWriterInner::Csv(csv_writer) => {
                csv_writer
                    .serialize(record)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            }
        }
        Ok(())
    }

    pub fn write_records(&mut self, records: &[FlowRecord]) -> io::Result<()> {
        for record in records {
            self.write_record(record)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        match &mut self.inner {
            FlowWriterInner::Jsonl(writer) => writer.flush(),
            FlowWriterInner::Csv(csv_writer) => csv_writer.flush(),
        }
    }

    pub fn format(&self) -> OutputFormat {
        self.format
    }
}

fn forward_flow_to_backend(addr: SocketAddr, json_bytes: &[u8]) {
    if let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(50)) {
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(100)));
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(100)));
        let prefix = b"{\"flow\":";
        let suffix = b"}";
        let body_len = prefix.len() + json_bytes.len() + suffix.len();
        let header = format!(
            "POST /api/classify HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n",
            addr,
            body_len
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(prefix);
        let _ = stream.write_all(json_bytes);
        let _ = stream.write_all(suffix);
        let _ = stream.flush();
        let mut resp_buf = [0u8; 64];
        let _ = std::io::Read::read(&mut stream, &mut resp_buf);
    }
}
