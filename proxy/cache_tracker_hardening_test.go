package proxy

import (
	"testing"
	"time"
)

// TestPromptCacheEvictsLRUWhenOverCapacity verifies the entries map is bounded:
// once it exceeds maxPromptCacheEntries, the least-recently-hit entries are
// evicted down to the cap.
func TestPromptCacheEvictsLRUWhenOverCapacity(t *testing.T) {
	tr := newPromptCacheTracker(time.Hour)
	now := time.Now()

	total := maxPromptCacheEntries + 5
	for i := 0; i < total; i++ {
		var fp [32]byte
		fp[0] = byte(i)
		fp[1] = byte(i >> 8)
		fp[2] = byte(i >> 16)
		tr.entries[fp] = promptCacheEntry{
			ExpiresAt: now.Add(time.Hour),
			TTL:       time.Hour,
			LastHit:   now.Add(time.Duration(i) * time.Second), // i=0 oldest
		}
	}

	tr.mu.Lock()
	tr.evictLRULocked()
	tr.mu.Unlock()

	if len(tr.entries) != maxPromptCacheEntries {
		t.Fatalf("expected %d entries after eviction, got %d", maxPromptCacheEntries, len(tr.entries))
	}
	// The 5 oldest (i=0..4) must be the ones evicted.
	for i := 0; i < 5; i++ {
		var fp [32]byte
		fp[0] = byte(i)
		fp[1] = byte(i >> 8)
		fp[2] = byte(i >> 16)
		if _, ok := tr.entries[fp]; ok {
			t.Fatalf("expected least-recently-hit entry %d to be evicted", i)
		}
	}
}

// TestSplitAgainstTotalAnchorsToRealTotal verifies the cache split is rescaled
// onto the real upstream input total while preserving the accounting identity
// input + creation + read == realTotal.
func TestSplitAgainstTotalAnchorsToRealTotal(t *testing.T) {
	// Estimate domain: covered = 800 (creation 500 + read 300), estTotal = 1000.
	u := promptCacheUsage{
		CacheCreationInputTokens:   500,
		CacheReadInputTokens:       300,
		CacheCreation5mInputTokens: 500,
	}
	got := u.splitAgainstTotal(1000, 2000)

	// ratio = 800/1000 = 0.8 → cacheTotal = 1600; read = 1600*300/800 = 600; creation = 1000.
	if got.CacheReadInputTokens != 600 {
		t.Fatalf("cache_read = %d, want 600", got.CacheReadInputTokens)
	}
	if got.CacheCreationInputTokens != 1000 {
		t.Fatalf("cache_creation = %d, want 1000", got.CacheCreationInputTokens)
	}
	billed := billedClaudeInputTokens(2000, got)
	if billed != 400 {
		t.Fatalf("billed input = %d, want 400", billed)
	}
	if sum := billed + got.CacheCreationInputTokens + got.CacheReadInputTokens; sum != 2000 {
		t.Fatalf("identity broken: input+creation+read = %d, want 2000", sum)
	}
}

// TestSplitAgainstTotalNoCoverageIsAllInput verifies that with no cache
// coverage the whole real total is billed as fresh input.
func TestSplitAgainstTotalNoCoverageIsAllInput(t *testing.T) {
	got := promptCacheUsage{}.splitAgainstTotal(1000, 2000)
	if got.CacheReadInputTokens != 0 || got.CacheCreationInputTokens != 0 {
		t.Fatalf("expected no cache tokens, got read=%d creation=%d", got.CacheReadInputTokens, got.CacheCreationInputTokens)
	}
	if billed := billedClaudeInputTokens(2000, got); billed != 2000 {
		t.Fatalf("expected all 2000 billed as input, got %d", billed)
	}
}
