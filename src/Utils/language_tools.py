import numpy as np

alphabet = '$ ,-.0123456789\\abcdefghijklmnopqrstuvwxyzßàâäåæçèéêëîïôöøùûüýþšžñ'


def random_jitter(
        sentence,
        sentence_length=None,
        alphabet=alphabet,
        expected_change=15,
):
    # ideally we want to change 1/15 of the sentence
    # so we have to repeat the while loop for len(s)/15 times
    # so the ratio should be 15*ln(2)/len(s)
    while (np.random.rand() > expected_change * np.log(2) / (len(sentence) + 1e-8)):
        if (np.random.rand() < 1 / 4):
            sentence = add_random_char(sentence, alphabet)
        elif (np.random.rand() < 1 / 3):
            sentence = remove_random_char(sentence)
        elif (np.random.rand() < 1 / 2):
            sentence = replace_random_char(sentence, alphabet)
        else:
            sentence = repeat_random_char(sentence)

    if sentence_length is not None:
        sentence = fix_sentence_length(sentence, sentence_length)
    return sentence


def fix_sentence_length(
        sentence,
        sentence_length,
):
    if len(sentence) > sentence_length:
        sentence = sentence[:sentence_length]
    elif len(sentence) < sentence_length:
        extra_space = sentence_length - len(sentence)
        sentence += ''.join([' '] * extra_space)
    return sentence


def add_random_char(
        sentence,
        alphabet=alphabet,
):
    pos = np.random.randint(len(sentence) + 1)
    random_char = alphabet[np.random.randint(len(alphabet))]
    return sentence[:pos] + random_char + sentence[pos:]


def remove_random_char(sentence):
    if len(sentence) < 5:
        return sentence
    pos = np.random.randint(len(sentence))
    return sentence[:pos] + sentence[pos + 1:]


def replace_random_char(
        sentence,
        alphabet=alphabet,
):
    pos = np.random.randint(len(sentence))
    random_char = alphabet[np.random.randint(len(alphabet))]
    return sentence[:pos] + random_char + sentence[pos + 1:]


def repeat_random_char(sentence):
    pos = np.random.randint(len(sentence))
    return sentence[:pos] + sentence[pos] + sentence[pos:]
