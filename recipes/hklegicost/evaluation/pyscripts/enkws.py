import sys, codecs, math, pickle, unicodedata, re
from collections import Counter

# Given two STM files (for SCALE'23), one a reference translation, and
# one a system output, we want to score how well we find the English
# words in the reference.

# Paul McNamee 7/24/23

def rmpunct(str):
    return re.sub('[\(\)\.,\'\"!\?!:;]', '', str)

def rmtrailpunct(str):
    return re.sub('-$', '', str)

def tidyup(str):
    return re.sub('&amp;', '&', str)

def onespace(str):
    return re.sub('\s+', ' ', str)

# Normalize text
def strnormalize(string):
    return onespace(rmpunct(re.sub('_', ' ', tidyup(string)).lower().strip())).strip()

# Stopword table from http://www.dcs.gla.ac.uk/idom/ir_resources/linguistic_utils/stop_words

stopterms = ['a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already','also','although','always','am','among','amongst','amoungst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','back','be','became','because','become','becomes','becoming','been','before','beforehand','behind','being','below','beside','besides','between','beyond','bill','both','bottom','but','by','call','can','cannot','cant','co','computer','con','could','couldnt','cry','de','describe','detail','do','done','down','due','during','each','eg','eight','either','eleven','else','elsewhere','empty','enough','etc','even','ever','every','everyone','everything','everywhere','except','few','fifteen','fify','fill','find','fire','first','five','for','former','formerly','forty','found','four','from','front','full','further','get','give','go','had','has','hasnt','have','he','hence','her','here','hereafter','hereby','herein','hereupon','hers','herself','him','himself','his','how','however','hundred','i','ie','if','in','inc','indeed','interest','into','is','it','its','itself','keep','last','latter','latterly','least','less','ltd','made','many','may','me','meanwhile','might','mill','mine','more','moreover','most','mostly','move','much','must','my','myself','name','namely','neither','never','nevertheless','next','nine','no','nobody','none','noone','nor','not','nothing','now','nowhere','of','off','often','on','once','one','only','onto','or','other','others','otherwise','our','ours','ourselves','out','over','own','part','per','perhaps','please','put','rather','re','same','see','seem','seemed','seeming','seems','serious','several','she','should','show','side','since','sincere','six','sixty','so','some','somehow','someone','something','sometime','sometimes','somewhere','still','such','system','take','ten','than','that','the','their','them','themselves','then','thence','there','thereafter','thereby','therefore','therein','thereupon','these','they','thick','thin','third','this','those','though','three','through','throughout','thru','thus','to','together','too','top','toward','towards','twelve','twenty','two','un','under','until','up','upon','us','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','will','with','within','without','would','yet','you','your','yours','yourself','yourselves']

stoptbl = set(stopterms)


# -----------------------------------------------------------------
def slurp(fname):
    lines = []
    f = open(fname, 'r')
    try:
        lines = f.readlines()
    finally:
        f.close()
    return lines

def wordsfromSTMline(line):
    line2 = ' '.join(line.split()[6:])
    line2 = re.sub(r"(\w)'(\w)", "\\1 \\2", line2)
    words = line2.split()
    words = filter(lambda x: not (x[0] == '[' and x[-1] == ']'), words)
    words = map(strnormalize, words)
    words = map(rmtrailpunct, words)
    words = filter(lambda x: len(x) > 0, words)
    words = filter(lambda x: not (len(x) == 1 and x.isalpha()), words)
    words = filter(lambda x: x not in stoptbl, words)
    words = set(words)
    #print("Line: %s\n  Words: %s\n" % (line, words))
    return words

def getsets(afile):
    lst = []
    for line in slurp(afile):
        lst.append( wordsfromSTMline(line))
    return lst

if __name__ == '__main__':  
    reffile = sys.argv[1]
    hypfile = sys.argv[2]
    (refsets, hypsets) = (getsets(reffile), getsets(hypfile))
    print("Len ref: %s / Len hyp: %s\n" % (len(refsets), len(hypsets)))
    inref = Counter()
    corr = Counter()
    inpred = Counter()
    for (ref, hyp) in zip(refsets, hypsets):
        #print("%s / %s\n" % (ref, hyp))
        for w in ref:
            inref[w] += 1
            if w in hyp:
                corr[w] += 1
        for w in hyp:
            inpred[w] += 1

    totalp = 0
    totalr = 0
    totalf = 0
    macro = 0
    for w in sorted(inref.keys()):  # Macro average
        p = corr[w] / max(inpred[w], 1e-8)
        totalp += p
        r = corr[w] / max(inref[w], 1e-8)
        totalr += r
        f1 = 2 * p * r / max((p + r), 1e-8)
        totalf += f1
        macro += f1
    print("Macro averaged p: %.4f, r: %.4f, f1: %.4f" % (totalp / len(inref), totalr / len(inref), totalf / len(inref)))

    micro = 0
    totalcorr = sum(corr.values())
    totalref = sum(inref.values())
    totalpred = sum(inpred.values())
    p = totalcorr / totalpred
    r = totalcorr / totalref
    f1 = 2 * p * r / (p + r)
    print("Micro averaged p: %.4f, r: %.4f, f1: %.4f" % (p, r, f1))
    
# end o' file
