# Licensing

`rigidity` is dual-licensed. The same code is available under two licences,
and you choose the one that fits what you are doing with it.

Copyright (C) 2026 Perfilev Dmitrii <dmitrii.perfilev2020@gmail.com>

## 1. GNU AGPL-3.0 — free, and free for good

The full text is in [`LICENSE`](LICENSE). SPDX: `AGPL-3.0-only`.

Free of charge, with no registration and nothing to sign, for **students,
university courses, theses, academic and public research, personal
projects, evaluation and non-profit organisations**. This is the licence
the project is developed under and the one almost every reader of this
file wants.

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

## 2. A commercial licence — for closed products

Buy this if you want to use `rigidity` in a product or service whose source
you do not intend to publish. It removes the copyleft obligation entirely;
everything else about the software is the same code.

Typical cases: embedding the registration pipeline in commercial metrology,
robotics or surveying software; a hosted service that registers customers'
scans; an OEM integration shipped to your own customers.

Terms are negotiated per case (per-seat, per-product or per-site, with or
without support and priority fixes). Write to
**dmitrii.perfilev2020@gmail.com** with what you are building and roughly how
many people or units are involved.

## Which one do I need?

| What you are doing | Licence |
|---|---|
| University course, thesis, published research | AGPL — free |
| Personal or hobby project | AGPL — free |
| Evaluating it before deciding | AGPL — free |
| Internal use in a company, results never distributed | AGPL — free |
| Open-source product, itself under AGPL-3.0 | AGPL — free |
| Closed-source product that ships to customers | commercial |
| SaaS or any hosted service | commercial |
| Redistributing under your own terms, or sublicensing | commercial |

If you are unsure which line you are on, ask before you build on it. The
answer is usually short and usually "the free one".

## Contributing

Dual licensing only works while one party holds the rights to the whole
work. A contribution accepted into this repository must therefore come with
a licence grant to the copyright holder wide enough to relicense it —
otherwise the commercial licence could not be offered for the file you
touched. Until a formal CLA exists, patches are accepted on the
understanding that you grant that right; say so in the pull request.

## Note

This file explains the arrangement in plain language. Where it differs from
[`LICENSE`](LICENSE) or from a signed commercial agreement, those documents
govern.
