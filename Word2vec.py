import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

set(stopwords.words('english'))
  
text_example = "This is a text example, showing off the stop words filtration. We work on this project, which is a very interesting project, this year. This is about sentiment analysis."
  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(text_example) 
  
output_text = [w for w in word_tokens if not w in stop_words] 
  
output_text = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        output_text.append(w) 
  
print("Texte avec stop words : "+str(word_tokens)) 
print("Texte sans stop words : "+str(output_text))