import nnsight


nnsight.CONFIG.set_default_api_key('api key')
nnsight.CONFIG.API.HOST = 'localhost:5001'
nnsight.CONFIG.API.SSL = False

#make sure you set HF_TOKEN in the NDIF server and in CLI since this is a gated model
model = nnsight.LanguageModel('meta-llama/Meta-Llama-3.1-8B')

with model.trace('The CN Tower is located in ', remote=True):
    output = model.output.save()

print(output)