# microgpt.ts
A conversion of Karpathy's MicroGPT to Typescript

[Andrej Karpathy](https://karpathy.ai/) gave us MicroGPT ([blog](https://karpathy.github.io/2026/02/12/microgpt/), [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)), a self contained, impressively minimalist implementation of a GPT.

Not everybody has Python as a preferred language, so, I decided to convert it to Typescript.

As I understand it, Karpathy's purpose was to help make this easier to learn.  I'm just trying to bring his work to a wider reading audience.

`TODO LINKS`: try, code, docs, build, CI runs

&nbsp;

## How to read this
There are two ways to read this.

1. You could do this Karpathy-style.  There is [a single-file TS GPT](https://github.com/StoneCypher/microgpt.ts/blob/main/src/one_file_ts/microgpt.ts) in this repository.
2. You could do this community-style.  There is a [modular GPT](https://github.com/StoneCypher/microgpt.ts/blob/main/src/ts/index.ts) in this repository too.

Or you could

&nbsp;

### What's the difference?

It's around 450 lines of code.  Chewing that down in one piece really suits a lot of people.  Other people don't want to think about, say, the random number generator while they're trying to learn Adam, or whatever.

The community style just takes the code and breaks it up into a couple of files - one for the tokenizer, one for the randomizer, one for autograd, et cetera - then adds docblock comments and tests.