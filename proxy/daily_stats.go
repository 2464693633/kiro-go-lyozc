package proxy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"kiro-go/config"
	"kiro-go/logger"
)

// DailyModelEntry aggregates one model's token usage and estimated cost for a
// single calendar day (UTC). Real fields come from the upstream; Sim fields
// come from the usage reported to the downstream client after multipliers.
type DailyModelEntry struct {
	Model string `json:"model"`

	// Upstream real token counts
	RealInput    int64 `json:"realInput"`
	RealOutput   int64 `json:"realOutput"`
	// Calculated real cost (USD) applying realCostMultiplier
	RealCostUSD float64 `json:"realCostUSD"`

	// Client-facing simulated token counts
	SimInput         int64 `json:"simInput"`
	SimOutput        int64 `json:"simOutput"`
	SimCacheRead     int64 `json:"simCacheRead"`
	SimCacheCreation int64 `json:"simCacheCreation"`
	// Calculated simulated cost (USD) using Anthropic model prices
	SimCostUSD float64 `json:"simCostUSD"`

	Requests int64 `json:"requests"`
}

// DailyStats holds all model entries for a single UTC day.
type DailyStats struct {
	Date   string                      `json:"date"` // "2006-01-02"
	Models map[string]*DailyModelEntry `json:"models"`
}

// dailyStatsStore manages in-memory accumulation and disk persistence.
type dailyStatsStore struct {
	mu      sync.Mutex
	days    map[string]*DailyStats // key = "2006-01-02"
	dataDir string
	stopCh  chan struct{}
	once    sync.Once
}

var globalDailyStats *dailyStatsStore

// InitDailyStats creates the singleton store and starts the background flush.
func InitDailyStats(dataDir string) {
	globalDailyStats = &dailyStatsStore{
		days:    make(map[string]*DailyStats),
		dataDir: dataDir,
		stopCh:  make(chan struct{}),
	}
	globalDailyStats.load()
	go globalDailyStats.flushLoop()
}

// CloseDailyStats flushes pending data and stops the background goroutine.
func CloseDailyStats() {
	if globalDailyStats == nil {
		return
	}
	globalDailyStats.once.Do(func() { close(globalDailyStats.stopCh) })
	globalDailyStats.flush()
}

// RecordDailyStats records one successful request's token usage into today's bucket.
func RecordDailyStats(model string, realInput, realOutput, simInput, simOutput, simCacheRead, simCacheCreation int) {
	if globalDailyStats == nil {
		return
	}
	globalDailyStats.record(model, realInput, realOutput, simInput, simOutput, simCacheRead, simCacheCreation)
}

// GetDailyStatsAll returns all stored daily stats ordered newest-first (up to 30 days).
func GetDailyStatsAll() []*DailyStats {
	if globalDailyStats == nil {
		return nil
	}
	return globalDailyStats.snapshot()
}

// --- model price table (USD per 1M tokens) ---

type modelPrice struct {
	Input    float64
	Output   float64
	Creation float64
	Read     float64
}

var modelPriceTable = []struct {
	key   string
	price modelPrice
}{
	{"claude-opus-4", modelPrice{15, 75, 18.75, 1.50}},
	{"claude-sonnet-4", modelPrice{3, 15, 3.75, 0.30}},
	{"claude-haiku-4", modelPrice{0.8, 4, 1.0, 0.08}},
	{"claude-opus-3", modelPrice{15, 75, 18.75, 1.50}},
	{"claude-sonnet-3", modelPrice{3, 15, 3.75, 0.30}},
	{"claude-haiku-3", modelPrice{0.25, 1.25, 0.3, 0.03}},
	{"claude", modelPrice{3, 15, 3.75, 0.30}},
}

func lookupPrice(model string) (modelPrice, bool) {
	lower := model
	for i := 0; i < len(lower); i++ {
		if lower[i] >= 'A' && lower[i] <= 'Z' {
			b := []byte(lower)
			for j := range b {
				if b[j] >= 'A' && b[j] <= 'Z' {
					b[j] += 32
				}
			}
			lower = string(b)
			break
		}
	}
	for _, row := range modelPriceTable {
		if len(lower) >= len(row.key) {
			for i := 0; i <= len(lower)-len(row.key); i++ {
				if lower[i:i+len(row.key)] == row.key {
					return row.price, true
				}
			}
		}
	}
	return modelPrice{}, false
}

func calcRealCostUSD(input, output int64, p modelPrice, multiplier float64) float64 {
	const M = 1e6
	return (float64(input)*p.Input+float64(output)*p.Output) / M * multiplier
}

func calcSimCostUSD(simInput, simOutput, simRead, simCreation int64, p modelPrice) float64 {
	const M = 1e6
	return (float64(simInput)*p.Input + float64(simOutput)*p.Output +
		float64(simRead)*p.Read + float64(simCreation)*p.Creation) / M
}

// --- internal methods ---

func (s *dailyStatsStore) record(model string, realIn, realOut, simIn, simOut, simRead, simCreation int) {
	date := time.Now().UTC().Format("2006-01-02")
	p, hasPrice := lookupPrice(model)
	realMul := config.GetRealCostMultiplier()

	s.mu.Lock()
	day, ok := s.days[date]
	if !ok {
		day = &DailyStats{Date: date, Models: make(map[string]*DailyModelEntry)}
		s.days[date] = day
	}
	entry, ok := day.Models[model]
	if !ok {
		entry = &DailyModelEntry{Model: model}
		day.Models[model] = entry
	}
	entry.RealInput += int64(realIn)
	entry.RealOutput += int64(realOut)
	entry.SimInput += int64(simIn)
	entry.SimOutput += int64(simOut)
	entry.SimCacheRead += int64(simRead)
	entry.SimCacheCreation += int64(simCreation)
	entry.Requests++
	if hasPrice {
		entry.RealCostUSD = calcRealCostUSD(entry.RealInput, entry.RealOutput, p, realMul)
		entry.SimCostUSD = calcSimCostUSD(entry.SimInput, entry.SimOutput, entry.SimCacheRead, entry.SimCacheCreation, p)
	}
	s.mu.Unlock()
}

func (s *dailyStatsStore) snapshot() []*DailyStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Collect up to 30 days, newest first
	type kv struct{ k string; v *DailyStats }
	var pairs []kv
	for k, v := range s.days {
		pairs = append(pairs, kv{k, v})
	}
	// Simple insertion sort (at most 30 entries)
	for i := 1; i < len(pairs); i++ {
		for j := i; j > 0 && pairs[j].k > pairs[j-1].k; j-- {
			pairs[j], pairs[j-1] = pairs[j-1], pairs[j]
		}
	}
	if len(pairs) > 30 {
		pairs = pairs[:30]
	}
	result := make([]*DailyStats, len(pairs))
	for i, p := range pairs {
		// deep copy
		cp := &DailyStats{Date: p.v.Date, Models: make(map[string]*DailyModelEntry)}
		for k, e := range p.v.Models {
			m := *e
			cp.Models[k] = &m
		}
		result[i] = cp
	}
	return result
}

func (s *dailyStatsStore) filePath() string {
	return filepath.Join(s.dataDir, "daily_stats.json")
}

func (s *dailyStatsStore) load() {
	data, err := os.ReadFile(s.filePath())
	if err != nil {
		return
	}
	var loaded map[string]*DailyStats
	if json.Unmarshal(data, &loaded) == nil {
		s.days = loaded
	}
}

func (s *dailyStatsStore) flush() {
	s.mu.Lock()
	data, err := json.Marshal(s.days)
	s.mu.Unlock()
	if err != nil {
		return
	}
	tmp := s.filePath() + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		logger.Warnf("[DailyStats] flush write failed: %v", err)
		return
	}
	if err := os.Rename(tmp, s.filePath()); err != nil {
		logger.Warnf("[DailyStats] flush rename failed: %v", err)
	}
}

func (s *dailyStatsStore) flushLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.flush()
		case <-s.stopCh:
			return
		}
	}
}
