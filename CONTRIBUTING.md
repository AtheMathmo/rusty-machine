# Contributing to rusty-machine

First of all thank you for your interest! I'm very keen to get more contributors onboard and am excited to help out in whichever ways I can. This is an early stage, developed-too-fast, library which could really benefit from more contributors.

Contributing can take place in many forms, including but not limited to:

- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Pull Requests](#pull-requests)
	- [Interested (but confused)?](#interested-but-confused)
	- [How can I test this?](#how-can-i-test-out-this-project)

Bug Reports and Feature Requests are easy and the project is happily accepting them now. Please fire away!

As for Pull Requests I am excited to take on new contributors who can help with the code. Please see the section below about getting started.

---

## Bug Reports

If you're using rusty-machine and run into what you believe to be a bug. Then please create an [issue](https://guides.github.com/features/issues/) to let me know. Even if you're not confident this is a bug I'd prefer to hear about it!

In the [issue](https://guides.github.com/features/issues/) please include a description of the bug, and the conditions needed to replicate it. Minimal conditions would be preferred but I understand this can often be a lot of work. If you can provide an example of the code producing the bug this would be really handy too!

## Feature Requests

I strongly encourage feature requests! I'd love to get feedback and learn what the community wants to see next from this project. I have my own goals and planned features which can be seen in the [Development](DEVELOPMENT.md) document. Even if a feature is listed here please feel free to request it regardless - it may affect the order in which I implement things.

To request a feature please open an [issue](https://guides.github.com/features/issues/) with a description of the feature requested. If you can include some technical details and requirements this would be a big help.

## Pull Requests

This section will cover the process for making code contributions to rusty-machine. Please feel free to make suggestions on how to improve this process (an issue on the repository will be fine).

### Getting Started

We currently use a [fork](https://help.github.com/articles/fork-a-repo/) and [pull request](https://help.github.com/articles/using-pull-requests/) model to allow contributions to rusty-machine.

Please take a look through the code and [API documentation](https://athemathmo.github.io/rusty-machine/rusty-machine/doc/rusty_machine/index.html) to identify the areas you'd like to help out with. Take a look through the current issues and see if there's anything you'd like to tackle. Simple issues will be tagged with the label `easy`.

If you decide you want to tackle an issue please comment on that issue stating that you would like to work on it. This will help us keep track of who is working on what. (I'm sure there's a better way to handle this - other ideas are welcome).

### Making Code Changes

So by now you should have the project forked and are ready to start working on the code. There are no hard conventions in place at the moment but please follow these general guidelines:

- Document all public facing functions, structs, fields, etc. You can check this by adding `#![deny(missing_docs)]` to the top of the `lib.rs` file. This should include examples, panics and failures. (If you see these missing anywhere in the current code please create an issue.)
- Add comments to all private functions detailing what they do.
- Make lots of small commits as opposed to one large commit.
- Ensure that all existing (and new) tests pass for each commit.
- Add new tests for any new functionality you add. This means examples within the documentation and for large functionality add test cases within the tests directory (following the current structure - though this is likely to change).
- There is (currently) no strict format for commit messages. But please be descriptive about the functionality you have added - this is much easier if using small commits as above!

### Creating the PR

Once the issue has been resolved please create the PR from your fork into the `master` branch. In the comments please reference the issue that the PR addresses, something like: "This resolves issue #XXX".

Other contributors will then review and give feedback. Once accepted the PR will be merged.

---

### Interested but confused?

Even now the project is fairly large and a bit overwhelming to join. **I'm happy to help people aboard and will do my best to make the process smooth.**

For now I have no special measures in place to assist with this. Due to this I'm happy for potential contributors to create new issues detailing their interests and we can open a conversation about how you can help out. Additionally please feel free to comment on existing tickets detailing your interest and ask any questions you need to about how to proceed.

### How can I test out this project?

**There are now some examples in the repository!**

Where is the test data? How do I know the current algorithms even work? How can I test things I've implemented?

All good questions. At the moment rusty-machine doesn't offer built in support for data loading or vizualisation - though I'm looking to fix this. For now I have been using my own separate project to test the algorithms on some open source data sets. This has been messy on my part but I'd be more than happy to share this if it helps other contributors test out their work (and my work thus far).

Otherwise I'd encourage you to write your own small projects to test out the algorithms. There are some great datasets [here](https://archive.ics.uci.edu/ml/datasets.html) from UCI. I've done some testing with the [Wisconsin breast cancer data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). Please provide feedback based on your experience!
