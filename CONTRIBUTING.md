# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and we will try to give credit appropriately.

# Bug reports

Use the [issue tracker](https://github.com/Quansight/numpy-threading-extensions/issues).
Please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

# Documentation improvements

FastNumPy could always use more documentation, whether as part of the
official FastNumPy docs, in docstrings, or even on the web in blog posts,
articles, and such.

# Feature requests and feedback

The best way to send feedback is to file an issue on the issue tracker.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

# Development

To set up `numpy-threading-extensions` for local development:

1. Fork [numpy-threading-extensions](https://github.com/Quansight/numpy-threading-extensions)
   (look for the "Fork" button).
2. Clone your fork locally
   ```
   git clone git@github.com:YOURGITHUBNAME/numpy-threading-extensions.git
   ```

3. Create a branch for local development::
   ```
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

4. When you're done making changes run all the tests with 
   ```
   python setup.py build_ext --inplace
   python -m pip install pytest
   python -m pytest tests
   ```

5. Commit your changes and push your branch to GitHub::
   ```
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

6. Submit a pull request through the GitHub website.

### Pull Request Guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Update documentation when there's new API, functionality etc.
2. Add a note to `CHANGELOG.rst` about the changes.
3. Add yourself to `AUTHORS.rst`.

<sup>1</sup>If you don't have all the necessary python versions available
locally you can rely on CI - it will [run the
tests](https://travis-ci.org/Quansight/numpy-threading-extensions/pull_requests)
for each change you add in the pull request.

It will be slower though ...

### Tips

To run a subset of tests::
```
python -m pytest -k test_myfeature
```


