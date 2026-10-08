# Licensing

`rigidity` is free software under a single licence, the GNU AGPL-3.0. There
is no commercial licence and no other terms on offer.

Copyright (C) 2026 Perfilev Dmitrii <dmitrii.perfilev2020@gmail.com>

## GNU AGPL-3.0 — free, and free for good

The full text is in [`LICENSE`](LICENSE). SPDX: `AGPL-3.0-only`.

Free of charge, with no registration and nothing to sign, for everyone:
students, university courses, theses, academic and public research,
personal projects, evaluation, non-profit organisations and companies
alike.

What it asks in return: anything you build on `rigidity` and then
**distribute or expose over a network** must itself be released under
AGPL-3.0, with its source. Section 13 is the part that distinguishes the
AGPL from the GPL — putting the code behind a web service counts as
conveying it, so a hosted product built on `rigidity` has to publish its
source too.

Two consequences that catch people out with Rust specifically:

- **Linking is enough.** A crate that depends on `rigidity-core` is a
  derivative work, and the whole binary falls under the AGPL. There is no
  LGPL-style exemption here.
- **Internal use is not distribution.** Running `rigidity` inside your own
  company, on your own machines, without shipping the result to anyone and
  without offering it as a service, triggers nothing at all. The AGPL only
  begins to ask for things when the software reaches someone else.

## What that means in practice

| What you are doing | What the AGPL asks |
|---|---|
| University course, thesis, published research | nothing |
| Personal or hobby project | nothing |
| Evaluating it before deciding | nothing |
| Internal use in a company, results never distributed | nothing |
| Open-source product, itself under AGPL-3.0 | nothing further |
| Product that ships to customers | release it under AGPL-3.0, with source |
| SaaS or any hosted service | release it under AGPL-3.0, with source |

## Contributing

Contributions are accepted under the same licence as the project: by
opening a pull request you agree that your patch is released under
AGPL-3.0-only.

## Note

This file explains the licence in plain language. Where it differs from
[`LICENSE`](LICENSE), that document governs.
