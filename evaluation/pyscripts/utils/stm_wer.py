#!/usr/bin/env python
# This code is intended for use as the official scoring script for SCALE23. It
# is mostly just a wrapper around sclite, except for it parses text sot that it
# can score using the "Mixed/Mandarin Error Rate" as opposed to just the normal
# WER. This script assumes that all reference files are stored as STMs, and all
# hypothesis files produced are also in the STMs.
#
# The script will produce the standard scoring files that sclite produces:
#    .sys, .raw, .prf, .pra,
#

 

import argparse
from pathlib import Path
import re
import regex as rex
from subprocess import check_call, DEVNULL
import unicodedata

import string
import chinese_converter


VALID_CATEGORIES = ('Mc', 'Mn', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Nd', 'Zs')
# Keep Markings such as vowel signs, all letters, and decimal numbers 
VALID_CATEGORIES = ('Mc', 'Mn', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Nd', 'Zs')
#noise_pattern = rex.compile(r'\[[^] ]*\]', re.UNICODE)
noise_pattern = rex.compile(r'\[(breath|cough|coughs|coughing|laugh|mouthnoise|hesitation)]')
#[hesitation]
#[laugh]
#[mouthnoise]
apostrophe_pattern = rex.compile(r"(\w)'(\w)")
html_tags = rex.compile(r"(&[^ ;]*;)|(</?[iu]>)")
KEEP_LIST = [u'\u2019']


_unicode = u"\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_buckwalter = u"|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"
_backwardMap = {ord(b):a for a,b in zip(_buckwalter, _unicode)}


def _arabic_preprocess(p):
    lines = []
    with open(p, 'r', encoding='utf-8') as f:
        for l in f:
            try:
                path, channel, speaker, start, end, lbl, text = l.strip().split(None, 6)
            except ValueError:
                path, channel, speaker, start, end, lbl = l.strip().split(None, 5)
                text = ''

            text = arabic_data_cleaning(text)
            yield ' '.join([path, channel, speaker, start, end, lbl, text])


def from_buckwalter(s):
    return s.translate(_backwardMap)
    

def read_tsv(f):
    text_data = list()
    for line in f:
        if not line.strip():
            continue
        text_data.append(line.strip().split('\t'))
    return text_data


_preNormalize = u" \u0629\u0649\u0623\u0625\u0622"
_postNormalize = u" \u0647\u064a\u0627\u0627\u0627"
_toNormalize = {ord(b):a for a,b in zip(_postNormalize,_preNormalize)}


def normalize_text_(s):
    return s.translate(_toNormalize)
    

def normalize_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub(r"(أ){2,}", "ا", text)
    text = re.sub(r"(ا){2,}", "ا", text)
    text = re.sub(r"(آ){2,}", "ا", text)
    text = re.sub(r"(ص){2,}", "ص", text)
    text = re.sub(r"(و){2,}", "و", text)
    return text   


def remove_english_characters(text):
    return re.sub(r'[^\u0600-\u06FF\s]+', '', text)


def remove_diacritics(text):
    #https://unicode-table.com/en/blocks/arabic/
    return re.sub(r'[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+', '', text)


def remove_punctuations(text):
    """ This function  removes all punctuations except the verbatim """
    
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    all_punctuations = set(arabic_punctuations + english_punctuations) # remove all non verbatim punctuations
    
    for p in all_punctuations:
        if p in text:
            text = text.replace(p, ' ')
    return text


def remove_extra_space(text):
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+\.\s+', '.', text)
    return text


def remove_dot(text):
    words = text.split()
    res = []
    for word in words:
        word = re.sub('\.$','',word)
        if word.replace('.','').isnumeric():  # remove the dot if it is not part of a number 
            res.append(word)
    
        else:
            res.append(word)
      
    return " ".join(res)
      

def east_to_west_num(text):
    eastern_to_western = {"٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9","٪":"%","_":" ","ڤ":"ف","|":" "}
    trans_string = str.maketrans(eastern_to_western)
    return text.translate(trans_string)
    
    
def digit2num(text, dig2num=False):

    """ This function is used to clean numbers"""

    # search for numbers with spaces
    # 100 . 000 => 100.000

    res = re.search('[0-9]+\s\.\s[0-9]+', text) 
    if res != None:
        t = re.sub(r'\s', '', res[0])
        text = re.sub(res[0], t, text)

    # seperate numbers glued with words 
    # 3أشهر => 3 أشهر
    # من10الى15 => من 10 الى 15
    res = re.findall(r'[^\u0600-\u06FF\a-z]+', text) # search for digits
    for match in res:
        if match not in {'.',' '}:
            text = re.sub(match, " "+ match+ " ",text)
            text = re.sub('\s+', ' ', text)

    # transliterate numbers to digits
    # 13 =>  ثلاثة عشر

    if dig2num == True:
        words = araby.tokenize(text)
        for i in range(len(words)):
            digit = re.sub(r'[\u0600-\u06FF]+', '', words[i])
            if digit.isnumeric():
                sub_word = re.sub(r'[^\u0600-\u06FF]+', '', words[i])
                if number.number2text(digit) != 'صفر':
                    words[i] = sub_word + number.number2text(digit)
            else:
                pass

        return " ".join(words)
    else:
        return text


def seperate_english_characters(text):
    """
        This function separates the glued English and Arabic words 
    """
    text = text.lower()
    res = re.findall(r'[a-z]+', text) # search for english words
    for match in res:
        if match not in {'.',' '}:
            text = re.sub(match, " "+ match+ " ",text)
            text = re.sub('\s+', ' ', text)
    return text
        

def arabic_data_cleaning(text):
    text = remove_punctuations(text)
    text = east_to_west_num(text)
    text = seperate_english_characters(text)
    text = remove_diacritics(text)
    text = remove_extra_space(text)
    text = normalize_arabic(text)
    text = normalize_text_(text)
    text = digit2num(text, False)
    return text



def _filter(s):
    return unicodedata.category(s) in VALID_CATEGORIES or s in KEEP_LIST 


def normalize_space(c):
    if unicodedata.category(c) == 'Zs':
        return " "
    else:
        return c


def _normalize(line):
    if line.strip('- ') != '':
        line_parts = noise_pattern.sub("", line)
        line_parts = apostrophe_pattern.sub(r"\1\u2019\2", line_parts)
        line_parts = html_tags.sub('', line_parts)
        line_parts_new = []
        for lp in line_parts.split("[noise]"):
            line_parts_new.append(
                ''.join(
                    [i for i in filter(_filter, lp.strip().replace('-', ' '))] 
                )
            )
        joiner = ' ' + "[noise]" + ' '
        line_new = joiner.join(line_parts_new)
        line_new = rex.sub(r"\p{Zs}", lambda m: normalize_space(m.group(0)), line_new)
        line_new = rex.sub(r' +', ' ', line_new).strip().lower()
        return line_new
    else:
        return ''


def _reformat_mixed_mandarin_text(text):
    '''
        Given some text, find consecutive chunks of Mandarin characters and
        convert these portions into individual characters. For the
        non-Mandarin chunks, we keep everything as is. 
    '''
    curr_pos = 0
    transcript = []
    num_syllables_used = [] 
    # A syllable is arbitrarily defined as 2 non-mandarin chars or 1 mandarin
    for group in re.finditer(r'([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD]+\s*)+', text):
        # First with start with non-Mandarin
        group_start, group_end = group.start(), group.end()
        if group_start > curr_pos:
            non_mandarin = text[curr_pos:group_start].replace('{', '').replace('}', '')
            non_mandarin_words = non_mandarin.split()
            # One syllable per 2 chars. If there is just 1 char in the word it
            # gets one syllable.
            num_syllables_used.extend([max(1, len(w)//2) for w in non_mandarin_words])
            transcript.extend(non_mandarin_words)

        # Now we do Mandarin
        mandarin = text[group_start:group_end]
        mandarin = re.sub(
            r'([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD])',
            r'\1 ', mandarin
        )
        mandarin = re.sub(r' +', ' ', mandarin).split()
        # One syllable per mandarin char.
        num_syllables_used.extend([1]*len(mandarin))
        mandarin = [chinese_converter.to_simplified(char) for char in mandarin]
        transcript.extend(mandarin)
        curr_pos = group_end

    # Once we've reached the end of the last mandarin group, the rest is
    # non-mandarin
    if curr_pos < len(text):
        non_mandarin = text[curr_pos:].replace('{', '').replace('}', '')
        non_mandarin_words = non_mandarin.split()
        num_syllables_used.extend([max(1, len(w)//2) for w in non_mandarin_words])
        transcript.extend(non_mandarin_words)

    return transcript, num_syllables_used


def _parse_stm_line(l):
    '''
        Parse a line of the stm file. The format is 

        audiopath channel speaker start end label text

        In the event that there is not text (it is empty for the segment), then
        we catch that case. 
    '''
    try:
        path, channel, speaker, start, end, lbl, text = l.strip().split(None, 6)
    except ValueError:
        path, channel, speaker, start, end, lbl = l.strip().split(None, 5)
        text = ''
    recoid = path.replace("//", "/")
    text_norm = _normalize(text)
    text, _ = _reformat_mixed_mandarin_text(text)
    text_norm, _ = _reformat_mixed_mandarin_text(text_norm)
    text = ' '.join(text)
    text = re.sub(r"\-\-+", "\-", text)
    text_norm = ' '.join(text_norm)
    text_norm = re.sub(r"\-\-+", "\-", text_norm)
    return {
        'recoid': recoid,
        'channel': channel,
        'speaker': speaker,
        'start': float(start),
        'end': float(end),
        'text': text,
        'label': lbl,
        'text_norm': text_norm,
    }


def _segment_to_uniform_ctm(seg, normalize=False):
    '''
        Convert an stm segment into a ctm. We make a special case when it comes
        to mandarin characters.
    '''
    channel, recoid = seg['channel'], seg['recoid']
    
    text = seg['text_norm'] if normalize else seg['text']
    
    transcript, num_syllables_used = _reformat_mixed_mandarin_text(text)

    duration_per_syllable = (seg['end'] - seg['start']) / sum(num_syllables_used) 
    start = seg['start']
    for w, num_syllables in zip(transcript, num_syllables_used):
        dur = num_syllables * duration_per_syllable 
        yield [recoid, channel, start, dur, w]
        start += dur


def main(args):
    # Make the output scoring directory
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    recos, recos_norm = {}, {}
    # Read the hypothesis stm file and convert it to ctm for sclite scoring.
    # The ctm is fake in the sense that the time markings are all fake and come
    # from the predefined, reference segmentation.
    if args.arabic:
        with open(odir / 'hyp.arabic.stm', 'w', encoding='utf-8') as f:
            for l in _arabic_preprocess(args.stm_hyp):
                print(l, file=f)
        with open(odir / 'ref.arabic.stm', 'w', encoding='utf-8') as f:
            for l in _arabic_preprocess(args.stm_ref):
                print(l, file=f)
        args.stm_hyp = str(odir / 'hyp.arabic.stm')
        args.stm_ref = str(odir / 'ref.arabic.stm')

    with open(args.stm_hyp, 'r', encoding='utf-8') as fp:
        for l in fp:
            seg = _parse_stm_line(l)
            recoid = seg['recoid']
            if recoid not in recos:
                recos[recoid] = []
                recos_norm[recoid] = []
            if seg['text'] != '':
                for ctm in _segment_to_uniform_ctm(seg, normalize=False):
                    recos[recoid].append(ctm) 
            if seg['text_norm'] != '':
                for ctm in _segment_to_uniform_ctm(seg, normalize=True):
                    recos_norm[recoid].append(ctm)

    stm_recos = {}
    # Read in the reference stm file.
    with open(args.stm_ref, 'r', encoding='utf-8') as fp1, open(odir / 'ref.stm', 'w', encoding='utf-8') as fp2, open(odir / 'ref.norm.stm', 'w', encoding='utf-8') as fp3:
        for l in fp1:
            seg = _parse_stm_line(l)
            recoid = seg['recoid']
            if recoid not in stm_recos:
                stm_recos[recoid] = []
            stm_recos[recoid].append(seg)
             
        # Sort the reference stm file and apply the mixed mandarin text
        # normalization scheme to enable scoring with mixed error rate.
        # The mixed error rate is the same as WER where each mandarin character
        # is treated as a separate word.
        for recoid in sorted(stm_recos.keys()):
            for seg in sorted(stm_recos[recoid], key=lambda x: (x['channel'], x['start'])):
                r, c, s = seg['recoid'], seg['channel'], seg['speaker']
                b, e, l, t, tn = seg['start'], seg['end'], seg['label'], seg['text'], seg['text_norm']
                print(f'{r} {c} {s} {b:0.2f} {e:0.2f} {l} {t}', file=fp2)
                print(f'{r} {c} {s} {b:0.2f} {e:0.2f} {l} {tn}', file=fp3)

    with open(odir / 'hyp.ctm', 'w', encoding='utf-8') as fp:
        for recoid in sorted(recos.keys()):
            for ctm in sorted(recos[recoid], key=lambda x: (x[1], x[2])):
                print(f'{ctm[0]} {ctm[1]} {ctm[2]:0.2f} {ctm[3]:0.2f} {ctm[4]} 1.0', file=fp) 
    
    with open(odir / 'hyp.norm.ctm', 'w', encoding='utf-8') as fp:
        for recoid in sorted(recos_norm.keys()):
            for ctm in sorted(recos_norm[recoid], key=lambda x: (x[1], x[2])):
                print(f'{ctm[0]} {ctm[1]} {ctm[2]:0.2f} {ctm[3]:0.2f} {ctm[4]} 1.0', file=fp)
    
    cmd = [args.sclite] + f"-r {odir / 'ref.stm'} stm -h {odir / 'hyp.ctm'} ctm -O {args.odir} -o all".split()
    status = check_call(cmd, shell=False, stderr=DEVNULL, stdout=DEVNULL)
    
    cmd = [args.sclite] + f"-r {odir / 'ref.norm.stm'} stm -h {odir / 'hyp.norm.ctm'} ctm -O {args.odir} -o all".split()
    status = check_call(cmd, shell=False, stderr=DEVNULL, stdout=DEVNULL)

    cmd = f'grep Sum {odir / "hyp.ctm.sys"}'.split()
    status = check_call(cmd, shell=False, stderr=DEVNULL)
    cmd = f'grep Sum {odir / "hyp.norm.ctm.sys"}'.split()
    status = check_call(cmd, shell=False, stderr=DEVNULL)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        
        Converts an stm to a ctm by assigning fake time labels to each word in
        a segment by distributing the words uniformly over time:

        ./utils/stm_wer.py /home/hltcoe/mwiesner/kaldi/tools/sctk/bin/sclite ref.stm hyp.stm scoring_dir 
        """
    )
    parser.add_argument("sclite", help='path to sclite binary')
    parser.add_argument("stm_ref", help='the reference stm file')
    parser.add_argument("stm_hyp", help='the hypothesis stm file')
    parser.add_argument("odir", help='the output directory to be created where'
        ' all scoring outputs are dumped.'
    )
    parser.add_argument("--arabic", action="store_true", help="use a different "
        "normalization scheme specific for arabic that is consistent with "
        "prior iwslt results."
    )
    args = parser.parse_args()
    main(args)
