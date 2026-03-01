# microgpt.ts
A conversion of Karpathy's MicroGPT to Typescript

[Andrej Karpathy](https://karpathy.ai/) gave us MicroGPT ([blog](https://karpathy.github.io/2026/02/12/microgpt/), [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)), a self contained, impressively minimalist implementation of a GPT.

Not everybody has Python as a preferred language, so, I decided to convert it to Typescript.

As I understand it, Karpathy's purpose was to help make this easier to learn.  I'm just trying to bring his work to a wider reading audience.

`TODO LINKS`: try, code, docs, build, CI runs

&nbsp;

## How to read this
There are two ways to read this.

1. You could do this Karpathy-style.  There is [a single-file TS GPT](TODO) in this repository.
2. You could do this community-style.  There is a [modular GPT](TODO) in this repository too.

&nbsp;

### What's the difference?

It's around 450 lines of code.  Chewing that down in one piece really suits a lot of people.  Other people don't want to think about, say, the random number generator while they're trying to learn Adam, or whatever.

The community style just takes the code and breaks it up into a couple of files - one for the randomizer, one for ADAM, et cetera.

I provide it unified for the first group, and modularized for the second group.  The main difference is whether it's split up, and that the split up version has longer docblock documentation (in the single file version the docblock comments were overwhelming.)  Also, I changed the entrypoint name to `index` in the modular version, and added unit tests, because some people find reading unit tests helps them understand.