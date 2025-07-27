#!/bin/bash


HOSTS=("xx.xx.xx.xx" "xx.xx.xx.xx")

USER="xxxx"


if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo "[*] Generating SSH key..."
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
else
    echo "[*] SSH key already exists."
fi

for HOST in "${HOSTS[@]}"; do
    echo "[*] Copying key to $USER@$HOST"

    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    
    PUBKEY=$(cat ~/.ssh/id_rsa.pub)
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    
    echo "[+] Done with $HOST"
done

echo "[✅] SSH done! Try ssh $USER@${HOSTS[0]}"

echo "[*] Installing pdsh for parallel SSH execution..."
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pdsh/pdsh-2.29.tar.bz2 && tar -xf pdsh-2.29.tar.bz2
cd pdsh-2.29 && ./configure --with-ssh --enable-static-modules --prefix=/usr/local && make && make install
pdsh -V
cd ..
