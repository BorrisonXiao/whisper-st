#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, os
from dataclasses import dataclass

@dataclass
class Ctm:
    wavbase: str
    chan: str # "A", "B", "1", "2"
    stt: float
    dur: float
    word: str
    conf: float = None
    def __str__(self):
        if self.conf is None:
            return f'{self.wavbase} {self.stt:.3f} {self.dur:.3f} {self.word}'
        else:
            return f'{self.wavbase} {self.stt:.3f} {self.dur:.3f} {self.word} {self.conf:.8f}'

@dataclass
class Stm:
    wavfile: str
    chan: str
    spkr: str
    stt: float
    ent: float
    label: str
    trans: str
    wavbase: str = None
    def __str__(self):
        return f'{self.wavfile} {self.chan} {self.spkr} {self.stt:.3f} {self.ent:.3f} {self.label} {self.trans}'
    def __post_init__(self):
        self.wavbase = os.path.splitext(os.path.basename(self.wavfile))[0]

def load_ctm(fname: str):
    ctm = {}
    n_elem = 0
    n_file_chan = 0
    with open(fname) as ifp:
        for line in ifp:
            e = line.split()
            assert len(e) == 5 or len(e) == 6, \
                f'ERROR: expect 5 or 6 elements in each line of CTM file {fname}: {line.strip()}'
            wavbase, chan, stt, dur, word = e[:5]
            stt = float(stt)
            dur = float(dur)
            conf = None if len(e) == 5 else e[5]
            c = Ctm(wavbase, chan, stt, dur, word, conf)
            if not c.wavbase in ctm:
                ctm[c.wavbase] = {}
            if not c.chan in ctm[c.wavbase]:
                ctm[c.wavbase][c.chan] = []
                n_file_chan += 1
            ctm[c.wavbase][c.chan].append(c)
            n_elem += 1
    for wavbase in ctm:
        for chan in ctm[wavbase]:
            dat = ctm[wavbase][chan]
            ctm[wavbase][chan] = sorted(dat, key = lambda x: (x.wavbase, x.chan, x.stt))
    print(f'Loaded CTM file {fname} with {n_elem} elements and {n_file_chan} file/channel pairs')
    return ctm

def load_stm(fname: str):
    stm = {}
    n_segs = 0
    n_file_chan = 0
    with open(fname) as ifp:
        for line in ifp:
            tmp = re.match(r'^([^<>]+<[^\s>]+>)\s*(.*)$', line)
            assert tmp is not None, f'ERROR: could not parse line from STM file {fname}: {line.strip()}'
            stminf, trans = tmp.groups()
            e = stminf.split()
            assert len(e) == 6, f'ERROR: STM info should have 6 elements in file {fname}: {line.strip()}'
            wavfile, chan, spkr, stt, ent, label = e
            stt = float(stt)
            ent = float(ent)
            s = Stm(wavfile, chan, spkr, stt, ent, label, trans)
            if not s.wavbase in stm:
                stm[s.wavbase] = {}
            if not s.chan in stm[s.wavbase]:
                stm[s.wavbase][s.chan] = []
                n_file_chan += 1
            stm[s.wavbase][s.chan].append(s)
            n_segs += 1
    for wavbase in stm:
        for chan in stm[wavbase]:
            dat = stm[wavbase][chan]
            stm[wavbase][chan] = sorted(dat, key = lambda x: (x.wavbase, x.chan, x.stt))
            # check for overlapping segments - that's a bug in the reference...
            prev_ent = None
            for s in stm[wavbase][chan]:
                if prev_ent is None:
                    prev_ent = s.ent
                else:
                    ovlp = prev_ent - s.stt
                    if ovlp > 0.0001:
                        print(f'WARNING: overlap detected in STM segments (wavbase={wavbase}, chan={chan}, overlap={ovlp:.3f}, prev_ent={prev_ent:.3f}, stt={s.stt:.3f})')
                    prev_ent = s.ent

    print(f'Loaded STM file {fname} with {n_segs} segments and {n_file_chan} file/channel pairs')
    return stm

def main():
    parser = argparse.ArgumentParser(description='Convert ASR CTM output to STM format using reference STM.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_ctm_hyp_file", type=str, help="Input CTM hypothesis file.")
    parser.add_argument("input_stm_ref_file", type=str, help="Input STM reference file.")
    parser.add_argument("output_stm_hyp_file", type=str, help="Output STM hypothesis file.")
    args = parser.parse_args()

    ctm = load_ctm(args.input_ctm_hyp_file)
    stm = load_stm(args.input_stm_ref_file)
    ret_stm = []
    n_ref_stm_seg = 0
    for wavbase in stm:
        if not wavbase in ctm:
            print(f'WARNING: wavbase "{wavbase}" in STM does not occur in CTM')
            continue
        stmw = stm[wavbase]
        ctmw = ctm[wavbase]
        for chan in stmw:
            if not chan in ctmw:
                print(f'WARNING: chan "{chan}" for wavbase "{wavbase}" in STM does not occur in CTM')
                continue
            stmwc = stmw[chan]
            ctmwc = ctmw[chan]
            n_ref_stm_seg += len(stmwc)
            for s in stmwc:
                trans = []
                for c in ctmwc:
                    ctm_stt = c.stt
                    ctm_ent = c.stt + c.dur
                    if ctm_stt > s.ent:
                        # Stop if we are beyond the end of this STM segment
                        break
                    if ctm_ent < s.stt:
                        # Skip if we are not yet inside the start of this STM segment
                        continue
                    ovlp = min(s.ent, ctm_ent) - max(s.stt, ctm_stt)
                    if ovlp >= 0.5 * c.dur:
                        # We'll assign ths word if at least half the duration is inside the segment:
                        # Not sure if this will actually keep us from assigning a CTM element to more 
                        # than one STM segment...
                        trans.append(c.word)
                    
                trans = ' '.join(trans)
                ret_s = Stm(s.wavfile, s.chan, s.spkr, s.stt, s.ent, s.label, trans)
                ret_stm.append(ret_s)
    assert len(ret_stm) == n_ref_stm_seg, f'ERROR: output STM only has {len(ret_stm)} segments while reference STM has {n_ref_stm_seg} segments'
    print(f'Created hypthesis STM file with {len(ret_stm)} segments')
    with open(args.output_stm_hyp_file, 'w') as ofp:
        for s in ret_stm:
            ofp.write(str(s) + '\n')
        
if __name__ == '__main__':
    main()
    
