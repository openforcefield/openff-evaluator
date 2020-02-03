#!/bin/sh

# Decrypt the file
mkdir $HOME

# --batch to prevent interactive command --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$OE_LICENSE_PASSPHRASE" \
--output $OE_LICENSE oe_license.gpg