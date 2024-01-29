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
# https://github.com/Alir3z4/stop-words/blob/bd8cc1434faeb3449735ed570a4a392ab5d35291/russian.txt
stopterms = ['a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already','also','although','always','am','among','amongst','amoungst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','back','be','became','because','become','becomes','becoming','been','before','beforehand','behind','being','below','beside','besides','between','beyond','bill','both','bottom','but','by','call','can','cannot','cant','co','computer','con','could','couldnt','cry','de','describe','detail','do','done','down','due','during','each','eg','eight','either','eleven','else','elsewhere','empty','enough','etc','even','ever','every','everyone','everything','everywhere','except','few','fifteen','fify','fill','find','fire','first','five','for','former','formerly','forty','found','four','from','front','full','further','get','give','go','had','has','hasnt','have','he','hence','her','here','hereafter','hereby','herein','hereupon','hers','herself','him','himself','his','how','however','hundred','i','ie','if','in','inc','indeed','interest','into','is','it','its','itself','keep','last','latter','latterly','least','less','ltd','made','many','may','me','meanwhile','might','mill','mine','more','moreover','most','mostly','move','much','must','my','myself','name','namely','neither','never','nevertheless','next','nine','no','nobody','none','noone','nor','not','nothing','now','nowhere','of','off','often','on','once','one','only','onto','or','other','others','otherwise','our','ours','ourselves','out','over','own','part','per','perhaps','please','put','rather','re','same','see','seem','seemed','seeming','seems','serious','several','she','should','show','side','since','sincere','six','sixty','so','some','somehow','someone','something','sometime','sometimes','somewhere','still','such','system','take','ten','than','that','the','their','them','themselves','then','thence','there','thereafter','thereby','therefore','therein','thereupon','these','they','thick','thin','third','this','those','though','three','through','throughout','thru','thus','to','together','too','top','toward','towards','twelve','twenty','two','un','under','until','up','upon','us','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','will','with','within','without','would','yet','you','your','yours','yourself','yourselves', \
'а','в','г','е','ж','и','к','м','о','с','т','у','я','бы','во','вы','да','до','ее','ей','ею','её','же','за','из','им','их','ли','мы','на','не','ни','но','ну','нх','об','он','от','по','со','та','те','то','ту','ты','уж','без','был','вам','вас','ваш','вон','вот','все','всю','вся','всё','где','год','два','две','дел','для','его','ему','еще','ещё','или','ими','имя','как','кем','ком','кто','лет','мне','мог','мож','мои','мой','мор','моя','моё','над','нам','нас','наш','нее','ней','нем','нет','нею','неё','них','оба','она','они','оно','под','пор','при','про','раз','сам','сих','так','там','тем','тех','том','тот','тою','три','тут','уже','чем','что','эта','эти','это','эту','алло','буду','будь','бывь','была','были','было','быть','вами','ваша','ваше','ваши','ведь','весь','вниз','всем','всех','всею','года','году','даже','двух','день','если','есть','зато','кого','кому','куда','лишь','люди','мало','меля','меня','мимо','мира','мной','мною','мочь','надо','нами','наша','наше','наши','него','нему','ниже','ними','один','пока','пора','пять','рано','сама','сами','само','саму','свое','свои','свою','себе','себя','семь','стал','суть','твой','твоя','твоё','тебе','тебя','теми','того','тоже','тому','туда','хоть','хотя','чаще','чего','чему','чтоб','чуть','этим','этих','этой','этом','этот','более','будем','будет','будто','будут','вверх','вдали','вдруг','везде','внизу','время','всего','всеми','всему','всюду','давно','даром','долго','друго','жизнь','занят','затем','зачем','здесь','иметь','какая','какой','когда','кроме','лучше','между','менее','много','могут','может','можно','можхо','назад','низко','нужно','одной','около','опять','очень','перед','позже','после','потом','почти','пятый','разве','рядом','самим','самих','самой','самом','своей','своих','сеаой','снова','собой','собою','такая','также','такие','такое','такой','тобой','тобою','тогда','тысяч','уметь','часто','через','чтобы','шесть','этими','этого','этому','близко','больше','будете','будешь','бывает','важная','важное','важные','важный','вокруг','восемь','всегда','второй','далеко','дальше','девять','десять','должно','другая','другие','других','другое','другой','занята','занято','заняты','значит','именно','иногда','каждая','каждое','каждые','каждый','кругом','меньше','начала','нельзя','нибудь','никуда','ничего','обычно','однако','одного','отсюда','первый','потому','почему','просто','против','раньше','самими','самого','самому','своего','сейчас','сказал','совсем','теперь','только','третий','хорошо','хотеть','хочешь','четыре','шестой','восьмой','впрочем','времени','говорил','говорит','девятый','десятый','кажется','конечно','которая','которой','которые','который','которых','наверху','наконец','недавно','немного','нередко','никогда','однажды','посреди','сегодня','седьмой','сказала','сказать','сколько','слишком','сначала','спасибо','человек','двадцать','довольно','которого','наиболее','недалеко','особенно','отовсюду','двадцатый','миллионов','несколько','прекрасно','процентов','четвертый','двенадцать','непрерывно','пожалуйста','пятнадцать','семнадцать','тринадцать','двенадцатый','одиннадцать','пятнадцатый','семнадцатый','тринадцатый','шестнадцать','восемнадцать','девятнадцать','одиннадцатый','четырнадцать','шестнадцатый','восемнадцатый','девятнадцатый','действительно','четырнадцатый','многочисленная','многочисленное','многочисленные','многочисленный',\
'a','al','algo','algunas','algunos','ante','antes','como','con','contra','cual','cuando','de','del','desde','donde','durante','e','el','ella','ellas','ellos','en','entre','era','erais','eran','eras','eres','es','esa','esas','ese','eso','esos','esta','estaba','estabais','estaban','estabas','estad','estada','estadas','estado','estados','estamos','estando','estar','estaremos','estará','estarán','estarás','estaré','estaréis','estaría','estaríais','estaríamos','estarían','estarías','estas','este','estemos','esto','estos','estoy','estuve','estuviera','estuvierais','estuvieran','estuvieras','estuvieron','estuviese','estuvieseis','estuviesen','estuvieses','estuvimos','estuviste','estuvisteis','estuviéramos','estuviésemos','estuvo','está','estábamos','estáis','están','estás','esté','estéis','estén','estés','fue','fuera','fuerais','fueran','fueras','fueron','fuese','fueseis','fuesen','fueses','fui','fuimos','fuiste','fuisteis','fuéramos','fuésemos','ha','habida','habidas','habido','habidos','habiendo','habremos','habrá','habrán','habrás','habré','habréis','habría','habríais','habríamos','habrían','habrías','habéis','había','habíais','habíamos','habían','habías','han','has','hasta','hay','haya','hayamos','hayan','hayas','hayáis','he','hemos','hube','hubiera','hubierais','hubieran','hubieras','hubieron','hubiese','hubieseis','hubiesen','hubieses','hubimos','hubiste','hubisteis','hubiéramos','hubiésemos','hubo','la','las','le','les','lo','los','me','mi','mis','mucho','muchos','muy','más','mí','mía','mías','mío','míos','nada','ni','no','nos','nosotras','nosotros','nuestra','nuestras','nuestro','nuestros','o','os','otra','otras','otro','otros','para','pero','poco','por','porque','que','quien','quienes','qué','se','sea','seamos','sean','seas','seremos','será','serán','serás','seré','seréis','sería','seríais','seríamos','serían','serías','seáis','sido','siendo','sin','sobre','sois','somos','son','soy','su','sus','suya','suyas','suyo','suyos','sí','también','tanto','te','tendremos','tendrá','tendrán','tendrás','tendré','tendréis','tendría','tendríais','tendríamos','tendrían','tendrías','tened','tenemos','tenga','tengamos','tengan','tengas','tengo','tengáis','tenida','tenidas','tenido','tenidos','teniendo','tenéis','tenía','teníais','teníamos','tenían','tenías','ti','tiene','tienen','tienes','todo','todos','tu','tus','tuve','tuviera','tuvierais','tuvieran','tuvieras','tuvieron','tuviese','tuvieseis','tuviesen','tuvieses','tuvimos','tuviste','tuvisteis','tuviéramos','tuviésemos','tuvo','tuya','tuyas','tuyo','tuyos','tú','un','una','uno','unos','vosotras','vosotros','vuestra','vuestras','vue<stro','vuestros','y','ya','yo','él','éramos', \
'فى','في','كل','لم','لن','له','من','هو','هي','قوة','كما','لها','منذ','وقد','ولا','نفسه','لقاء','مقابل','هناك','وقال','وكان','نهاية','وقالت','وكانت','للامم','فيه','كلم','لكن','وفي','وقف','ولم','ومن','وهو','وهي','يوم','فيها','منها','مليار','لوكالة','يكون','يمكن','مليون','حيث','اكد','الا','اما','امس','السابق','التى','التي','اكثر','ايار','ايض','','ثلاثة','الذاتي','الاخيرة','الثاني','الثانية','الذى','الذي','الان','امام','ايام','خلال','حوالى','الذين','الاول','الاولى','بين','ذلك','دون','حول','حين','الف','الى','انه','ا','ل','ضمن','انها','جميع','الماضي','الوقت','المقبل','اليوم','ـ','ف','و','و6','قد','لا','ما','مع','مساء','هذا','واحد','واضاف','واضافت','فان','قبل','قال','كان','لدى','ن','و','هذه','وان','واكد','كانت','واوضح','مايو','ب','ا','أ','،','عشر','عدد','عدة','عشرة','عدم','عام','عاما','عن','عند','عندما','على','عليه','عليها','زيارة','سنة','سنوات','تم','ضد','بعد','بعض','اعادة','اعلنت','بسبب','حتى','اذا','احد','اثر','برس','باسم','غدا','شخصا','صباح','اطار','اربعة','اخرى','بان','اجل','غير','بشكل','حاليا','بن','به','ثم','اف','ان','او','اي','بها','صفر',\
'فى','في','كل','لم','لن','له','من','هو','هي','قوة','كما','لها','منذ','وقد','ولا','لقاء','مقابل','هناك','وقال','وكان','وقالت','وكانت','فيه','لكن','وفي','ولم','ومن','وهو','وهي','يوم','فيها','منها','يكون','يمكن','حيث','االا','اما','االتى','التي','اكثر','ايضا','الذى','الذي','الان','الذين','ابين','ذلك','دون','حول','حين','الى','انه','اول','انها','ف','و','و6','قد','لا','ما','مع','هذا','واحد','واضاف','واضافت','فان','قبل','قال','كان','لدى','نحو','هذه','وان','واكد','كانت','واوضح','ب','ا','أ','،','عن','عند','عندما','على','عليه','عليها','تم','ضد','بعد','بعض','حتى','اذا','احد','بان','اجل','غير','بن','به','ثم','اف','ان','او','اي','بها',\
]

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

def wordsfromSTMline(line, dostopping=False):
    line2 = ' '.join(line.split()[6:])
    line2 = re.sub(r"(\w)'(\w)", "\\1 \\2", line2)
    words = line2.split()
    words = filter(lambda x: not (x[0] == '[' and x[-1] == ']'), words)
    words = map(strnormalize, words)
    words = map(rmtrailpunct, words)
    words = filter(lambda x: len(x) > 0, words)
    words = filter(lambda x: not (len(x) == 1 and x.isalpha()), words)
    if dostopping:
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
    for arg in sys.argv:
        if re.search('help', arg):
            print("\npython kwseval.py REFSTM HYPSTM [trigraph (eng)] [filterk (1000)\n  REFSTM is a reference STM (that may or may not have been GLM'd)\n  HYPSTM is a system output STM that is line-aligned with the REFSTM file\nThe input STM files are presumed to use whitespace to separate tokens.\nMandarin doesn't have a stoplist.\n\nThe output is micro and macro-averaged precision, recall, and F1 scores")
            sys.exit(0)
    reffile = sys.argv[1]
    hypfile = sys.argv[2]
    (refsets, hypsets) = (getsets(reffile), getsets(hypfile))
    #print("Len ref: %s / Len hyp: %s\n" % (len(refsets), len(hypsets)))
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
