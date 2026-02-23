# Latin silver data 
TM --> Sources in the volltext are automatically extracted from the RE entry.
TM --> These sources are automatically expanded, so we know which text and, if possible, which paragraph they refer to.
Us --> check if the text is in the LASLA corpus, 
Us --> check if the paragraph is found.
If yes:
    Us --> check if the retrieved paragraph is more fine-grained than "book" (including books introduces too many errors).
    if yes: 
        Us --> Locate the retrieved paragraph in LASLA
        Us --> (LASLA tokens are linked to lila:lemmas and there are existing mappings between lilalemmas and trismegistos nam ids) find the tokens with lila:lemmas the map back to the RE Volltext's nam ids.
        Us --> (deviation from previous) --> Rule based method of creating multitoken entities (two subsequent tokens with the same RE-id assigned to it are a multitoken).
