use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::config::OutputFormat;
use crate::features::FlowRecord;

pub struct FlowWriter {
    format: OutputFormat,
    inner: FlowWriterInner,
}

enum FlowWriterInner {
    Jsonl(Box<dyn Write>),
    Csv(csv::Writer<Box<dyn Write>>),
}

impl FlowWriter {
    pub fn new(output_path: Option<&Path>, format: OutputFormat) -> io::Result<Self> {
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

        Ok(Self { format, inner })
    }

    pub fn write_record(&mut self, record: &FlowRecord) -> io::Result<()> {
        match &mut self.inner {
            FlowWriterInner::Jsonl(writer) => {
                serde_json::to_writer(&mut *writer, record)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                writer.write_all(b"\n")?;
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
