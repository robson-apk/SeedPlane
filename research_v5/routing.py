"""Routing contracts. No semantic or quality labels enter these routers."""
import hashlib,struct
import numpy as np
FIELDS=('request','generation','boundary','model_version','target','source')
MODES=('unfiltered','boundary_id','hadamard_v4','exact_envelope','seedplane_v5')
H=np.ones((1,1),dtype=np.float32)
while len(H)<32:H=np.block([[H,H],[H,-H]])
H/=np.sqrt(np.float32(32))
def complement(a,b):return float(np.clip(np.dot(H[a%32],H[b%32]),0,1))
def accept(mode,actual,expected,seen):
 if mode=='unfiltered':return True
 if mode=='boundary_id':return actual[2]==expected[2]
 if mode=='hadamard_v4':return complement(actual[2],expected[2])>.5
 if tuple(actual)!=tuple(expected):return False
 if mode=='seedplane_v5' and complement(actual[2],expected[2])<=.5:return False
 if actual in seen:return False
 seen.add(actual);return True

def wire_bytes(mode):
 # Six uint64 fields needed by exact protocol. V4 uses a 32-float code.
 # target transport framing/payload are common to all modes and excluded.
 return {'unfiltered':0,'boundary_id':8,'hadamard_v4':128,'exact_envelope':48,'seedplane_v5':176}[mode]
