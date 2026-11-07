def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	# Your code here
	np.random.seed(42)
	encoder, hparams, params = load_encoder_hparams_and_params()

	ids = encoder.encode(prompt)
	generated = []
	n_head = hparams['n_head']

	def linear(x, layer):
		return x @ layer['w'] + layer['b']

	def layer_norm(x, layer):
		mean = x.mean(axis = -1, keepdims=True)
		var = x.var(axis = -1, keepdims=True)
		return (x - mean) / np.sqrt(var + 1e-5) * layer['g'] + layer['b']
	
	def gelu(x):
		return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
	
	def softmax(x):
		exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return exp / exp.sum(axis=-1, keepdims=True)
	
	for _ in range(n_tokens_to_generate):
		L = len(ids)

		x = params['wte'][ids] + params['wpe'][:L]

		for block in params['blocks']:
			a = layer_norm(x, block['ln_1'])

			q, k, v = np.split(linear(a, block['attn']['c_attn']), 3, axis=-1)

			d_head = q.shape[-1] // n_head 

			q = q.reshape(L, n_head, d_head).transpose(1, 0, 2)
			k = k.reshape(L, n_head, d_head).transpose(1, 0, 2)
			v = v.reshape(L, n_head, d_head).transpose(1, 0, 2)

			scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)
			scores += np.triu(np.full((L,L), -1e10), 1)

			attn = softmax(scores) @ v 
			attn = attn.transpose(1, 0, 2).reshape(L,-1)

			x = x + linear(attn, block['attn']['c_proj'])
			y = layer_norm(x, block['ln_2'])

			x = x + linear(gelu(linear(y, block['mlp']['c_fc'])), block['mlp']['c_proj'])

		logits = layer_norm(x, params['ln_f']) @ params['wte'].T
		
		next_id = int(np.argmax(logits[-1]))

		ids.append(next_id)
		generated.append(next_id)

	return encoder.decode(generated)

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [{
			"mlp": {
				"c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
				"c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
			},
			"attn": {
				"c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
				"c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
			},
			"ln_1": {"g": np.ones(10), "b": np.zeros(10)},
			"ln_2": {"g": np.ones(10), "b": np.zeros(10)},
		}],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params