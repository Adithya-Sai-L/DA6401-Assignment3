In this assignment you will experiment with the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) released by Google. This dataset contains pairs of the following form: 

$x$.      $y$

ajanabee अजनबी.

i.e., a word in the native script and its corresponding transliteration in the Latin script (the way we type while chatting with our friends on WhatsApp etc). Given many such $(x_i, y_i)_{i=1}^n$ pairs your goal is to train a model $y = \hat{f}(x)$ which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 