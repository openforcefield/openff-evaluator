#!/bin/sh

# Decrypt the file
mkdir $HOME/secrets

# --batch to prevent interactive command --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$OE_LICENSE_PASSPHRASE" --output $HOME/secrets/oe_license.txt oe_license.txt.gpg