1. **Class balance.** **Observation:** both labels are well represented.
   **Evidence:** 1,003 spam and 942 ham comments
   (51.6% spam). **Interpretation:** accuracy should still be compared
   with a majority baseline; this curated sample does not estimate YouTube's real spam prevalence.
2. **Message length.** **Observation:** spam comments are longer on average.
   **Evidence:** spam averages 137.5 characters and
   22.1 words, versus ham's
   50.1 characters and 9.2 words.
   **Interpretation:** length offers useful descriptive context, but group means
   overlap and do not supply a reliable decision rule for every comment.
3. **Group vocabulary.** **Observation:** the most frequent lexical tokens differ.
   **Evidence:** spam's leading term is **check** (557 occurrences),
   while ham's is **song** (224 occurrences).
   **Interpretation:** language patterns motivate TF-IDF classification; a frequent
   word is not by itself proof that an individual message is spam.
4. **Links.** **Observation:** URL usage differs between the groups.
   **Evidence:** 19.0% of spam comments and
   1.2% of ham comments contain a detected URL.
   **Interpretation:** a URL marker can capture promotional behavior without
   memorizing domains; legitimate links and link-free spam still require context.
5. **Stop words.** **Observation:** common English function words occupy much of the text.
   **Evidence:** they account for 46.1% of the 30,821 tokens.
   **Interpretation:** removing them makes exploratory topic charts easier to read,
   but keeping them in the classifier permits useful phrase patterns.
6. **Source variation.** **Observation:** video groups have different spam proportions.
   **Evidence:** rates range from 47.4% for Shakira
   to 54.9% for Eminem.
   **Interpretation:** selecting whole videos can shift a sample's label mix;
   this motivates reporting sampling distributions and caution about unseen videos.
