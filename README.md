# microgpt.ts
A conversion of Karpathy's MicroGPT to Typescript

[Andrej Karpathy](https://karpathy.ai/) gave us MicroGPT ([blog](https://karpathy.github.io/2026/02/12/microgpt/), [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)), a self contained, impressively minimalist implementation of a GPT.

Not everybody has Python as a preferred language, so, I decided to convert it to Typescript.

As I understand it, Karpathy's purpose was to help make this easier to learn.  I'm just trying to bring his work to a wider reading audience.

* 💻 [Try it](https://stonecypher.github.io/microgpt.ts/usage.html), or
* 🔍 [Read the code](https://github.com/StoneCypher/microgpt.ts/blob/main/src/ts/index.ts), or
* 👓 [Read the docs](https://stonecypher.github.io/microgpt.ts/docs/), or 
* 🏗️ [See the builds]()

&nbsp;

## How to read this
There are two ways to read this.

1. You could do this Karpathy-style.  There is [a single-file TS GPT](https://github.com/StoneCypher/microgpt.ts/blob/main/src/one_file_ts/microgpt.ts) in this repository.
2. You could do this community-style.  There is a [modular GPT](https://github.com/StoneCypher/microgpt.ts/blob/main/src/ts/index.ts) in this repository too.

Or you could

&nbsp;

### What's the difference?

It's around 450 lines of code.  Chewing that down in one piece really suits a lot of people.  Other people don't want to think about, say, the random number generator while they're trying to learn Adam, or whatever.

The community style just takes the code and breaks it up into a couple of files - one for the tokenizer, one for the randomizer, one for autograd, et cetera - then adds docblock comments and tests.  Community style also adds a single function called `test` to `index`, which shows a person how these library calls are made.

&nbsp;

## How to use this

Well, generally use a production GPT.  This is correct, but it doesn't do any of the fancy stuff that gives you real speed.  However, for learning purposes, [we've included a web GUI that does this live](https://stonecypher.github.io/microgpt.ts/usage.html).  There's a copy of Karpathy's "tiny shakespeare" to train on in `src/html`.

Alternately, you can download it and run it locally.  If you do, please use `npm run local` to start a local webserver, then hit the page at [https://localhost:4400/usage.html](localhost:4400/usage.html), instead of loading locally from a `file` url; `file://` has CORS consequences, and you won't be able to upload your text trainer.

&nbsp;

## How to contribute

Patches, features, and bugfixes are accepted graciously.  It's easier to accept a PR that has a description, so I know what it does.  Please fork the repo and send a PR back from your fork.