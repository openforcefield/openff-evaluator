#!/bin/sh

# Decrypt the file
mkdir $HOME/secrets

# --batch to prevent interactive command --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$TEST_PASSPHRASE" --output $HOME/secrets/test.txt test.txt.gpg

echo $HOME/secrets/test.txt