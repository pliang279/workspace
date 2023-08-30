import wolframalpha

# Taking input from user
question = 'what is sin(30)'

# App id obtained by the above steps
app_id = '22PXAV-P9E8RP9H6U'

# Instance of wolf ram alpha 
# client class
client = wolframalpha.Client(app_id)

# Stores the response from 
# wolf ram alpha
res = client.query(question)

# Includes only text from the response
answer = next(res.results).text

print(answer)