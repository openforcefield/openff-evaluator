#!/bin/sh

echo $TEST_PATH

# --batch to prevent interactive command --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$TEST_PASSPHRASE" --output $TEST_PATH test.txt.gpg