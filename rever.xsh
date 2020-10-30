$PROJECT = 'fast-numpy-loops'
$ACTIVITIES = [
              'version_bump',  # Changes the version number in various source files (setup.py, __init__.py, etc)
              'changelog',  # Uses files in the news folder to create a changelog for release
              'tag',  # Creates a tag for the new version number
              'push_tag',  # Pushes the tag up to the $TAG_REMOTE
              'ghrelease'  # Creates a Github release entry for the new tag
               ]
$VERSION_BUMP_PATTERNS = [  # These note where/how to find the version numbers
                         ('src/fast_numpy_loops/__init__.py', r'__version__\s*=.*', "__version__ = '$VERSION'"),
                         ('setup.py', r'version\s*=.*,', "version='$VERSION',")
                         ]
$CHANGELOG_FILENAME = 'CHANGELOG.rst'  # Filename for the changelog
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'  # Filename for the news template
$CHANGELOG_HEADER = '.. current developments\n\n$VERSION\n====================\n\n',
$PUSH_TAG_REMOTE = 'git@github.com:Quansight/numpy-threading-extensions.git'  # Repo to push tags to

$GITHUB_ORG = 'Quansight'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'numpy-threading-extensions'  # Github repo for Github releases  and conda-forge 
$GHRELEASE_TARGET = 'v0.1'
