#!/usr/bin/env python3
import re
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("r", encoding="utf-8"), default="-")
    parser.add_argument("glm", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("format", choices=["ctm", "stm"])
    parser.add_argument("output", type=argparse.FileType("w", encoding="utf-8"), default="-")
    parser.add_argument("--keep-all-tokens", action="store_true")

    args = parser.parse_args()

    word_mappings = {}
    GLM_RE = re.compile(u'^\s?(%?\S+)\s+=>\s+((\S+)\s*((\S+)\s*)?)($|\/|;)', flags=re.UNICODE)
    with args.glm as f:
        for i, line in enumerate(f):
            if line[:2] == ';;':
                continue

            glm_match = GLM_RE.match(line.strip().upper())
            if glm_match is None:
                sys.stderr.write('glm line %s (%d) does not match regex\n' % (line.strip(), i+1))
            else:
                orig_word = glm_match.group(1).lower().replace("[","").replace("]","")
                mapped_word = glm_match.group(2).lower()
                mapped_word = mapped_word.replace("[","").replace("{","").replace("]","").replace("}","")
                word_mappings[orig_word] = mapped_word

                # glm might be missing '%' suffix on original word
                word_mappings[u'%%%s' % orig_word] = mapped_word

    with args.input as input_fp, args.output as output_fp:
        if args.format == 'ctm':
            for line in input_fp:
                parts = line.strip().split()
                if len(parts) < 5 or len(parts) > 6:
                    sys.stderr.write("Error bad line in CTM %s : %s\n" % (args.input, line.encode('utf-8')))
                    continue

                audio_filename = parts[0]
                side = parts[1]
                start = parts[2]
                duration = parts[3]
                word = parts[4]
                if len(parts) == 6:
                    conf = parts[5]
                else:
                    conf = 0.5

                if word.lower() in word_mappings:
                    cleaned_word = word_mappings[word.lower()]
                    if cleaned_word.lower() == u'%hesitation':
                        continue
                elif not args.keep_all_tokens and word[0] == '[' and word[-1] == ']':
                    continue
                elif word[0] == '-' or word[-1] == '-':
                    continue
                elif '~' in word:
                    cleaned_word = re.sub('~', '', word)
                else:
                    cleaned_word = word

                cleaned_word = cleaned_word.strip()

                if len(cleaned_word) == 0:
                    continue

                print('%s %s %s %s %s %s' % (audio_filename, side, start, duration, cleaned_word, conf), file=output_fp)

        elif args.format == 'stm':
            for line in input_fp:
                if line[:2] == ';;':
                    output_fp.write(line)
                    continue

                parts = line.strip().split()

                if len(parts) < 7:
                    sys.stderr.write("Error bad line in STM %s : %s\n" % (args.input, line.encode('utf-8')))
                    continue

                audio_filename = parts[0]
                side = parts[1]
                speaker_id = parts[2]
                start = parts[3]
                stop = parts[4]
                tags = parts[5]
                words = parts[6:]
                cleaned_words = []
                for word in words:
                    # lowercase word and remove some trailing punctuation   
                    normalized_word = word.lower()
                    normalized_word = normalized_word.rstrip('.,!')
                    if word[0] == '[' and word[-1] == ']':
                        cleaned_words.append('(' + word + ')')
                    elif word[0] == '-' or word[-1] == '-':
                        cleaned_words.append('(' + word + ')')
                    elif '~' in word:
                        cleaned_words.append(re.sub('~', '', word))
                    #elif word in word_mappings:
                    elif normalized_word in word_mappings:   
                        #cleaned_words.append('(' + word_mappings[word] + ')')
                        cleaned_words.append('(' + word_mappings[normalized_word] + ')')
                    else:
                        cleaned_words.append(word)

                text = ' '.join(cleaned_words)
                text = re.sub(' +', ' ', text)
                text = re.sub('^ ', '', text)
                text = re.sub(' $', '', text)
                text = text.strip()

                if len(text) == 0:
                    continue

                print('%s %s %s %s %s %s %s' % (audio_filename, side, speaker_id, start, stop, tags, text), file=output_fp)

if __name__ == '__main__':
    main()

