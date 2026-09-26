/// Numerically stable streaming statistical accumulator using Welford's algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingStats {
    count: u64,
    min: f64,
    max: f64,
    sum: f64,
    mean: f64,
    m2: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add a new observation in O(1) time and O(1) space.
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        let delta = value - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    #[inline]
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    #[inline]
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mean
        }
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let var = self.m2 / ((self.count - 1) as f64);
            if var < 0.0 {
                0.0
            } else {
                var
            }
        }
    }

    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Tracks inter-arrival times (IAT) between packets in microseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct IatTracker {
    last_timestamp: Option<u64>,
    stats: StreamingStats,
    total: u64,
}

impl Default for IatTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl IatTracker {
    pub fn new() -> Self {
        Self {
            last_timestamp: None,
            stats: StreamingStats::new(),
            total: 0,
        }
    }

    /// Update with incoming packet timestamp in microseconds.
    pub fn update(&mut self, timestamp_micros: u64) {
        if let Some(prev) = self.last_timestamp {
            let iat = if timestamp_micros >= prev {
                timestamp_micros - prev
            } else {
                0 // Clock skew or out-of-order within microsecond resolution
            };
            self.stats.add(iat as f64);
            self.total += iat;
        }
        self.last_timestamp = Some(timestamp_micros);
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.stats.count()
    }

    #[inline]
    pub fn min(&self) -> f64 {
        self.stats.min()
    }

    #[inline]
    pub fn max(&self) -> f64 {
        self.stats.max()
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.stats.mean()
    }

    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.stats.std_dev()
    }

    #[inline]
    pub fn total(&self) -> f64 {
        self.total as f64
    }

    #[inline]
    pub fn last_seen(&self) -> Option<u64> {
        self.last_timestamp
    }
}

/// Active / Idle burst tracker for CICIDS2017 features.
/// An Idle period is defined when inter-arrival time exceeds the idle threshold (default: 1.0s = 1,000,000 µs).
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveIdleTracker {
    idle_threshold_micros: u64,
    last_packet_time: Option<u64>,
    current_active_start: Option<u64>,
    current_active_end: Option<u64>,
    active_stats: StreamingStats,
    idle_stats: StreamingStats,
}

impl Default for ActiveIdleTracker {
    fn default() -> Self {
        Self::new(1_000_000) // 1 second default threshold for CICFlowMeter
    }
}

impl ActiveIdleTracker {
    pub fn new(idle_threshold_micros: u64) -> Self {
        Self {
            idle_threshold_micros,
            last_packet_time: None,
            current_active_start: None,
            current_active_end: None,
            active_stats: StreamingStats::new(),
            idle_stats: StreamingStats::new(),
        }
    }

    pub fn update(&mut self, timestamp_micros: u64) {
        if let Some(prev) = self.last_packet_time {
            let gap = if timestamp_micros >= prev {
                timestamp_micros - prev
            } else {
                0
            };

            if gap > self.idle_threshold_micros {
                // The gap was idle
                self.idle_stats.add(gap as f64);

                // Conclude previous active burst
                if let (Some(start), Some(end)) = (self.current_active_start, self.current_active_end) {
                    if end >= start {
                        self.active_stats.add((end - start) as f64);
                    }
                }

                // Start new active burst
                self.current_active_start = Some(timestamp_micros);
                self.current_active_end = Some(timestamp_micros);
            } else {
                // Continuation of current active burst
                self.current_active_end = Some(timestamp_micros);
            }
        } else {
            // First packet of the flow
            self.current_active_start = Some(timestamp_micros);
            self.current_active_end = Some(timestamp_micros);
        }

        self.last_packet_time = Some(timestamp_micros);
    }

    /// Finalize tracking when a flow terminates or expires.
    pub fn finalize(&mut self) {
        if let (Some(start), Some(end)) = (self.current_active_start.take(), self.current_active_end.take()) {
            if end >= start {
                self.active_stats.add((end - start) as f64);
            }
        }
    }

    #[inline]
    pub fn active_mean(&self) -> f64 {
        self.active_stats.mean()
    }

    #[inline]
    pub fn active_std(&self) -> f64 {
        self.active_stats.std_dev()
    }

    #[inline]
    pub fn active_max(&self) -> f64 {
        self.active_stats.max()
    }

    #[inline]
    pub fn active_min(&self) -> f64 {
        self.active_stats.min()
    }

    #[inline]
    pub fn idle_mean(&self) -> f64 {
        self.idle_stats.mean()
    }

    #[inline]
    pub fn idle_std(&self) -> f64 {
        self.idle_stats.std_dev()
    }

    #[inline]
    pub fn idle_max(&self) -> f64 {
        self.idle_stats.max()
    }

    #[inline]
    pub fn idle_min(&self) -> f64 {
        self.idle_stats.min()
    }
}
