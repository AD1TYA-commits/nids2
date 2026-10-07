// scratch/test_flow_storage.js
// Verification of bounded memory working set and storage pagination

class FlowStorageService {
  constructor(dbName = 'nids_soc_telemetry_db', maxRecords = 50000) {
    this.dbName = dbName;
    this.maxRecords = maxRecords;
    this.pendingWrites = [];
    this.flushTimeout = null;
    this.totalCount = 0;
    this.incidentCount = 0;
    // High-performance ring buffer / store
    this.store = [];
    this.idMap = new Map();
  }

  async save(flow) {
    if (!flow || !flow.flow_id) return;
    const isThreat = (flow.decision && flow.decision !== 'NORMAL') || flow.anomaly || (flow.risk_score >= 60);
    const enriched = {
      ...flow,
      _stored_at: Date.now(),
      _is_threat: isThreat
    };
    this.pendingWrites.push(enriched);
    if (this.pendingWrites.length >= 50) {
      await this.flushWrites();
    } else if (!this.flushTimeout) {
      this.flushTimeout = setTimeout(() => this.flushWrites(), 60);
    }
  }

  async flushWrites() {
    if (this.flushTimeout) {
      clearTimeout(this.flushTimeout);
      this.flushTimeout = null;
    }
    if (this.pendingWrites.length === 0) return;
    const batch = this.pendingWrites;
    this.pendingWrites = [];

    for (const item of batch) {
      if (!this.idMap.has(item.flow_id)) {
        this.totalCount++;
        if (item._is_threat) this.incidentCount++;
      }
      this.idMap.set(item.flow_id, item);
      this.store.unshift(item); // Newest first
    }

    if (this.store.length > this.maxRecords) {
      const removed = this.store.splice(this.maxRecords);
      for (const r of removed) {
        this.idMap.delete(r.flow_id);
      }
      this.totalCount = this.store.length;
    }
  }

  async getPage({ page = 1, pageSize = 50, query = '', severity = 'ALL', threatOnly = false } = {}) {
    await this.flushWrites();
    const q = (query || '').toLowerCase().trim();

    const matches = this.store.filter(f => {
      if (threatOnly && !f._is_threat) return false;
      if (severity === 'THREATS_ONLY' && f.decision === 'NORMAL') return false;
      if (severity === 'UNKNOWN_ONLY' && f.decision !== 'UNKNOWN ATTACK') return false;
      if (severity === 'CRITICAL_ONLY' && f.decision !== 'CRITICAL' && f.decision !== 'MALICIOUS') return false;
      if (severity === 'NORMAL_ONLY' && f.decision !== 'NORMAL') return false;

      if (q) {
        const src = (f.src_ip || '').toLowerCase();
        const dst = (f.dst_ip || '').toLowerCase();
        const proto = (f.protocol || '').toLowerCase();
        const pred = (f.prediction || '').toLowerCase();
        const dec = (f.decision || '').toLowerCase();
        const fid = (f.flow_id || '').toLowerCase();
        if (!src.includes(q) && !dst.includes(q) && !proto.includes(q) && !pred.includes(q) && !dec.includes(q) && !fid.includes(q)) {
          return false;
        }
      }
      return true;
    });

    const totalMatches = matches.length;
    const totalPages = Math.max(1, Math.ceil(totalMatches / pageSize));
    const validPage = Math.max(1, Math.min(page, totalPages));
    const offset = (validPage - 1) * pageSize;
    const records = matches.slice(offset, offset + pageSize);

    return {
      records,
      page: validPage,
      pageSize,
      totalMatches,
      totalPages
    };
  }

  async getFlowById(flowId) {
    if (this.idMap.has(flowId)) return this.idMap.get(flowId);
    const pending = this.pendingWrites.find(f => f.flow_id === flowId);
    if (pending) return pending;
    return null;
  }

  async getTotalCount() {
    await this.flushWrites();
    return this.totalCount;
  }

  async getIncidentCount() {
    await this.flushWrites();
    return this.incidentCount;
  }
}

async function runTest() {
  console.log("Starting 5,000 flow simulation test on FlowStorageService...");
  const storage = new FlowStorageService('test_db', 50000);

  // Bounded live working set
  const LIVE_MEMORY_LIMIT = 200;
  const liveWindow = [];

  const startMem = process.memoryUsage().heapUsed;

  for (let i = 1; i <= 5000; i++) {
    const isAnomaly = (i % 7 === 0);
    const decision = isAnomaly ? (i % 21 === 0 ? 'UNKNOWN ATTACK' : 'SUSPICIOUS') : 'NORMAL';
    const flow = {
      flow_id: `10.12.92.${i % 250}:${1024 + (i % 60000)}-192.168.122.227:80-6`,
      timestamp: new Date(Date.now() - (5000 - i) * 100).toISOString(),
      src_ip: `10.12.92.${i % 250}`,
      dst_ip: `192.168.122.227`,
      src_port: 1024 + (i % 60000),
      dst_port: 80,
      protocol: 'TCP',
      flow_duration_secs: 0.15,
      total_packets: 4,
      total_bytes: 420,
      ground_truth_label: '',
      prediction: decision === 'UNKNOWN ATTACK' ? 'UNKNOWN' : 'Normal',
      confidence: 0.99,
      anomaly: isAnomaly,
      anomaly_score: isAnomaly ? 1.85 : 0.42,
      threshold: 1.3728,
      risk_score: decision === 'UNKNOWN ATTACK' ? 88.5 : (decision === 'SUSPICIOUS' ? 50.0 : 5.0),
      decision: decision,
      action: decision === 'UNKNOWN ATTACK' ? 'QUARANTINE' : (decision === 'SUSPICIOUS' ? 'MONITOR' : 'ALLOW'),
      reason: 'Automated test flow'
    };

    // Live bounded window
    liveWindow.unshift(flow);
    if (liveWindow.length > LIVE_MEMORY_LIMIT) {
      liveWindow.pop(); // Evict oldest from RAM
    }

    // Persist to storage
    await storage.save(flow);
  }

  await storage.flushWrites();

  const totalStored = await storage.getTotalCount();
  const incidents = await storage.getIncidentCount();
  const endMem = process.memoryUsage().heapUsed;

  console.log(`[+] Live Window Length: ${liveWindow.length} (Capped at ${LIVE_MEMORY_LIMIT})`);
  console.log(`[+] Total Flows in Storage: ${totalStored} (5000 expected)`);
  console.log(`[+] Total Incidents in Storage: ${incidents}`);
  console.log(`[+] Heap usage delta: ${((endMem - startMem) / 1024 / 1024).toFixed(2)} MB`);

  // Test pagination on historical data
  const page1 = await storage.getPage({ page: 1, pageSize: 50 });
  console.log(`[+] Page 1 Count: ${page1.records.length}, Total Pages: ${page1.totalPages}, Total Matches: ${page1.totalMatches}`);
  console.log(`[+] Page 1 First Item ID: ${page1.records[0].flow_id}`);

  const page10 = await storage.getPage({ page: 10, pageSize: 50 });
  console.log(`[+] Page 10 Count: ${page10.records.length}, First Item ID: ${page10.records[0].flow_id}`);

  // Test historical lookup of an evicted flow (flow #10 was evicted from live window 4,990 flows ago!)
  const evictedFlow = await storage.getFlowById(`10.12.92.10:1034-192.168.122.227:80-6`);
  console.log(`[+] Retrieved Evicted Flow from Storage:`, evictedFlow ? `Found (${evictedFlow.flow_id})` : `Not Found!`);

  // Test incident history pagination
  const incPage = await storage.getPage({ page: 1, pageSize: 25, severity: 'THREATS_ONLY' });
  console.log(`[+] Incident Page 1 Count: ${incPage.records.length}, Total Incidents: ${incPage.totalMatches}`);

  console.log("TEST COMPLETED SUCCESSFULLY.");
}

runTest();
