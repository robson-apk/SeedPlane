// Layout of the sampling scratch buffer `sh` (uint words) shared by the top-k / top-p passes (smode == 2):
const uint H1C = 0u, H1M = 1024u, H2C = 2048u, H2M = 3072u;   // level-1 / level-2 counts and fixed-point masses
const uint S_B1 = 4096u, S_CNT0 = 4097u, S_MASS0 = 4098u, S_M = 4099u, S_Z = 4100u, SH_WORDS = 4104u;
const float FIX = 2147483648.0;                               // probability mass in units of 2^-31
const uint NB = 1024u;
// Selection criterion walking from the most likely token down: top-k -> count >= k, top-p only -> mass > p.
bool reached(uint cnt, uint mass, uint k, float p) { return k > 0u ? cnt >= k : float(mass) > p * FIX; }
