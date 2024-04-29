tinymongo is integrated into the data solution of this program, but it is no longer maintained and a tweak in another package broke it on later versions of Python.

This 'package' fixes the problem so that the program can continue to use tinymongo normally.

Any time you would use TinyMongoClient, import this version and use its TinyMongoClient instead.

See https://github.com/schapman1974/tinymongo/issues/58