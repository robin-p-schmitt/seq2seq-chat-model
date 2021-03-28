import re
from pathlib import Path
from datetime import datetime
import time
import os
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter

def prepare_whatsapp_data(directory, spell_dict = None):
    """Convert exported WhatsApp chats into desired format.
    
    Exported chats have the form:
    
    dd/mm/yy, hh:mm - <author>: <text>
    or
    <text> (if the sender used new lines in a message)
    """
    def get_date_from_msg(msg):
        """Get time of message in seconds.
        """
        date = re.findall(r"(\d+/\d+/\d+, \d+:\d+) -", msg)

        if date:
            return time.mktime(datetime.strptime(date[0], r"%m/%d/%y, %H:%M").timetuple())
        else:
            return 0
        
    def get_sorted_sequences(filename):
        """Get a sorted list of messages from exported chat txt file.
        
        Sometimes, the order of messages gets messed up during WhatsApp export.
        """
        f = open(filename, "r", encoding = "utf-8")
        messages = []
        index = 0
        for line in f:
            # if line starts with a date, append it to list of messages
            if re.match(r"\d+/\d+/\d+, \d+:\d+", line):
                messages.append(line)
                index += 1
            # otherwise, the line is a result of typing a new line and 
            # it is therefore appended to the last message
            else:
                messages[index - 1] += line
        # sort messages by time of receival
        messages.sort(key = get_date_from_msg)

        return messages
    
    def separate_emojis(text):
        """Separate multiple consecutive emojis with spaces.
        
        In mobile chat applications, emojis are often typed consecutively
        without spaces in-between.
        """
        # this matches emojis which are directly preceded by non-whitespace chars
        emoji_pattern = re.compile(r"(\S*?)" u"([\U00010000-\U0010ffff(\u2764\uFE0F)])")
        # insert blank in front of emoji
        text = re.sub(emoji_pattern, r"\1 \2", text)
        # this matches emojis which are directly succeeded by non-whitespace chars
        emoji_pattern = re.compile(u"([\U00010000-\U0010ffff(\u2764\uFE0F)])" r"(\S+)")
        # insert blank behind emoji
        text = re.sub(emoji_pattern, r"\1 \2", text)
        
        return text
    
    def replace_multichars(text):
        """Replace consecutive occurrences of the same character with only 2 occurrences.
        
        In chat applications, words are often stretched out. E.g.: "heeeey".
        This functions replaces such character series with two instances of the character.
        """
        new = text
        # find series of three or more consecutive chars
        multichars = re.findall(r'((\w)\2{3,})', text)
        if multichars:
            for char in multichars:
                # replace them with a sequence of two chars
                new = new.replace(char[0], char[1] * 2)

        return new     

    def replace_digits(text):
        """Replace any digit with "number"
        """
        text = re.sub(r"\d+", "number", text)
        
        return text
    
    for filename in Path(directory).glob("*.txt"):
        if os.path.basename(filename) == "word_correct.txt":
            continue
        # get head and tail of path
        split = os.path.split(filename)
        # new "edited" directory
        path = os.path.join(split[0], "edited")
        # create directory if not existant
        if not os.path.exists(path):
            os.mkdir(path)
        # start a new file for the current whatsapp chat
        with open(os.path.join(split[0], "edited", split[1]), "w+", encoding = "utf-8") as f:
            # get list of messages sorted by time
            messages = get_sorted_sequences(filename)
            # get list of tuples of author and message for every message
            messages = [(author, text.lower().strip()) for msg in messages for author, text in re.findall(r".+? - (.+?): (.+)", msg, flags = re.DOTALL)]
            # group consecutive messages of the same author
            groups = groupby(messages, key = itemgetter(0))
            # get list of grouped messages
            messages = [[msg.strip() for author, msg in group] for i, group in groups]
            # separate emojis with spaces
            messages = [[separate_emojis(msg) for msg in group] for group in messages]
            # replace 3 or more multiple consecutive characters with 2 of those chars
            messages = [[replace_multichars(msg) for msg in group] for group in messages]
            # replace digits with a special <number> token
            messages = [[replace_digits(msg) for msg in group] for group in messages]
            # always write two consecutive messages separated by a tab into the file
            for i in range(len(messages) - 1):
                f.write(str(messages[i]) + "\t" + str(messages[i + 1]) + "\n")
            
def prepare_cornell_movie_corpus(path_to_lines, path_to_dialogues):
    """Convert the cornell movie corpus to a common format.
    
    Args:
        path_to_lines: file which maps movie line ids to their corresponding text.
        path_to_dialogues: file with lists of line ids that correspond to a dialogue in a movie.
    """
    movie_lines = {}
    # create dictionary which maps line ids to their corresponding text
    # from the movie_lines file
    with open(os.path.join("data", path_to_lines), "r", encoding = 'iso-8859-1') as f:
        for line in f:
            line_id, _, _, _, text = line.split("+++$+++")
            line_id = line_id.strip()
            text = text.lower().strip()
            movie_lines[line_id] = text
    
    dialogues = []
    
    # parse the lists of dialogues from the dialogues file
    with open(os.path.join("data", path_to_dialogues), "r", encoding = 'iso-8859-1') as f:
        for line in f:
            _, _, _, dialogue = line.split("+++$+++")
            dialogue = eval(dialogue)
            dialogues.append(dialogue)
            
    # create new path for edited movie dialogues
    # base_path = os.path.split(path_to_lines)[0]
    new_file_path = os.path.join("data", "training", "movie-corpus-edited.txt")
    
    # write the dialogues to the new file by separating
    # answers and questions with a tab
    with open(new_file_path, "w+", encoding = "utf-8") as f:
        for dialogue in dialogues:
            for i in range(len(dialogue) - 1):
                line1 = movie_lines[dialogue[i]]
                line2 = movie_lines[dialogue[i + 1]]
                f.write(str([line1]) + "\t" + str([line2]) + "\n")

prepare_cornell_movie_corpus("cornell-movie-corpus/movie_lines.txt", "cornell-movie-corpus/movie_conversations.txt")
prepare_whatsapp_data("whatsapp")