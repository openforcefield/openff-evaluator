#!/bin/sh

echo $TEST_FILE

# --batch to prevent interactive command --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$TEST_PASSPHRASE" --output $TEST_FILE test.txt.gpg