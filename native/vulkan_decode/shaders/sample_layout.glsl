// Layout of the sampling scratch buffer `sh` (uint words) shared by the top-k / top-p passes (smode == 2):
const uint H1C = 0u, H1M = 1024u, H2C = 2048u, H2M = 3072u;   // level-1 / level-2 counts and fixed-point masses
const uint S_B1 = 4096u, S_CNT0 = 4097u, S_MASS0 = 4098u, S_M = 4099u, S_Z = 4100u, SH_WORDS = 4104u;
const float FIX = 2147483648.0;                               // probability mass in units of 2^-31
const uint NB = 1024u;
const uint MASS_GUARD = 262144u;                              // 2^-13: covers fixed-point + float32 Z error for V <= 262k
// Selection criterion walking from the most likely token down. Every fixed-point term was floored, so the true
// cumulative mass lies in [mass, mass + cnt). Use that conservative upper bound for top-p: it can stop by at most
// V*2^-31 mass early, but cannot drift into tokens below the reference support because of accumulated rounding loss.
bool reached(uint cnt, uint mass, uint k, float p) {
    return k > 0u ? cnt >= k : float(mass + cnt + MASS_GUARD) > p * FIX;
}
