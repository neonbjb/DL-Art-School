# Pretends to be the stylegan2 Network class for intercepting pickle load requests.
# Horrible hack. Please don't judge me.

# Globals for storing these networks because I have no idea how pickle is doing this internally.
generator, discriminator, gen_ema = {}, {}, {}

class Network:
    def __setstate__(self, state: dict) -> None:
        global generator, discriminator, gen_ema
        name = state['name']
        if name in ['G_synthesis', 'G_mapping', 'G', 'G_main']:
            if name != 'G' and name not in generator.keys():
                generator[name] = state
            else:
                gen_ema[name] = state
        elif name in ['D']:
            discriminator[name] = state
