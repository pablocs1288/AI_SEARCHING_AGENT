
import os
import logging
import spacy

from spacy.tokens import Span
from spacy.language import Language


from src.tools.tool import Tool


@Language.component("board_member_detector")
def board_member_detector(doc):
    board_keywords = {"ceo", "founder", "chairman", "board", "director"}
    new_ents = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and any(k in ent.sent.text.lower() for k in board_keywords):
            # Replace with BOARD_MEMBER
            new_ent = Span(doc, ent.start, ent.end, label=doc.vocab.strings["BOARD_MEMBER"])
            new_ents.append(new_ent)
        else:
            # Keep original
            new_ents.append(ent)

    doc.ents = new_ents
    return doc


class NERTool(Tool):

    def __init__(self):
        super().__init__()

    def invoke_tool(self, text: str) -> str:
        names = self._extract_names(text)
        if names:
            names = list(set(names)) # removes duplicates
            return ", ".join(name for name in names)
        
        return  "No Board Members Found"

        
    # this strategy could be better
    def _extract_names(self, text):
        """ Extract likely names from text using spaCy NER """
        nlp = spacy.load(os.environ['LOCAL_NER_MODEL']) # sm -> small, md -> medium, lg -> large dslim/bert-base-NER may be a better option as it was fined-tuned for PERSON and ROLES ()

        nlp.get_pipe("ner").add_label("BOARD_MEMBER")
        nlp.add_pipe("board_member_detector", after="ner")

        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "BOARD_MEMBER"]
        #return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
