"""Text preprocessing shared by MoodTune training and inference."""

import re
import unicodedata


_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[^\w\s]", re.IGNORECASE)
_NEGATORS = {
    "not", "no", "never", "neither", "nor", "n't", "dont", "don't",
    "doesnt", "doesn't", "didnt", "didn't", "isnt", "isn't", "arent",
    "aren't", "wasnt", "wasn't", "werent", "weren't", "cannot", "can't",
    "cant", "couldnt", "couldn't", "wouldnt", "wouldn't", "shouldnt",
    "shouldn't", "wont", "won't",
}
_NEGATION_WINDOW = 3
_NEGATED_POSITIVE_RE = re.compile(
    r"\b(?:not|never|no|dont|don't|doesnt|doesn't|didnt|didn't|isnt|isn't|"
    r"arent|aren't|wasnt|wasn't|werent|weren't|cannot|can't|cant|couldnt|"
    r"couldn't|wouldnt|wouldn't|shouldnt|shouldn't|wont|won't)"
    r"(?:\W+\w+){0,3}\W+(?:happy|joyful|glad|cheerful|pleased|excited)\b",
    re.IGNORECASE,
)
_OTHER_EMOTION_CUE_RE = re.compile(
    r"\b(?:sad|sadness|lonely|heartbroken|angry|anger|mad|furious|afraid|"
    r"scared|fearful|worried|anxious|surprised|shocked|amazed|love|loving)\b",
    re.IGNORECASE,
)


def mark_negated_words(text: str) -> str:
    """Prefix words within a short negation scope with ``neg_``.

    For example, ``I am not happy`` becomes ``i am not neg_happy``. This lets
    TF-IDF and the classifier learn a separate feature for negated wording,
    instead of treating ``happy`` as an ordinary positive cue.
    """
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = _TOKEN_RE.findall(normalized)
    remaining_scope = 0
    output = []

    for token in tokens:
        if token in _NEGATORS:
            remaining_scope = _NEGATION_WINDOW
            output.append(token)
            continue

        if not token.isalpha():
            remaining_scope = 0
            output.append(token)
            continue

        if remaining_scope > 0:
            output.append("neg_" + token)
            remaining_scope -= 1
        else:
            output.append(token)

    return " ".join(output)


def predict_emotion_id(text, vectorizer, classifier) -> int:
    """Predict with the classifier plus a narrow negation fallback.

    The six-class dataset has no neutral label. If the text explicitly negates
    positive wording (for example, "not happy") and contains no other clear
    emotion cue, use sadness as the closest available negative-valence class.
    Otherwise, leave the classifier prediction unchanged.
    """
    match = _NEGATED_POSITIVE_RE.search(text)
    if match:
        remaining_text = text[:match.start()] + " " + text[match.end():]
        if not _OTHER_EMOTION_CUE_RE.search(remaining_text):
            return 0  # LABEL_NAMES[0] == sadness

    features = vectorizer.transform([text])
    return int(classifier.predict(features)[0])
