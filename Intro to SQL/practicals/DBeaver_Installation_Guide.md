# DBeaver Installation Guide

**DBeaver** is a free, open-source database tool that works with virtually any database — MySQL, PostgreSQL, SQLite, Oracle, and more. Follow the instructions for your operating system below.

---

## Before You Begin

- Make sure you have a stable internet connection
- You will need administrator/sudo privileges on your machine
- Minimum: 4 GB RAM, 500 MB free disk space
- Java is **bundled** with the installer — you do not need to install it separately

---

## Download DBeaver

Go to the official download page:

> **https://dbeaver.io/download/**

Download the **Community Edition (free)** for your operating system.

---

## Installation Instructions

### Windows

1. Download the **Windows installer** (`.exe`) from the download page.
2. Double-click the downloaded `.exe` file.
3. If prompted by User Account Control, click **Yes**.
4. Follow the setup wizard:
   - Click **Next** through the welcome screen.
   - Accept the license agreement.
   - Choose the installation folder (the default is fine).
   - Click **Install**.
5. Click **Finish** when the installation completes.
6. Launch DBeaver from the Start Menu or the desktop shortcut.

---

### macOS

1. Download the **macOS DMG** file from the download page.
2. Open the downloaded `.dmg` file.
3. Drag the **DBeaver** icon into your **Applications** folder.
4. Open your **Applications** folder and double-click **DBeaver**.
5. If you see a warning about an unidentified developer:
   - Go to **System Settings → Privacy & Security**
   - Scroll down and click **Open Anyway**
6. DBeaver will launch successfully.

---

### Linux

#### Option A — Debian/Ubuntu (.deb)

1. Download the **Linux Debian package** (`.deb`) from the download page.
2. Open a terminal and run:

   ```bash
   sudo dpkg -i dbeaver-ce_<version>_amd64.deb
   ```

   Replace `<version>` with the version number in your downloaded filename.

3. If you see dependency errors, fix them with:

   ```bash
   sudo apt-get install -f
   ```

4. Launch DBeaver from your applications menu, or run:

   ```bash
   dbeaver
   ```

#### Option B — Red Hat/Fedora/CentOS (.rpm)

1. Download the **Linux RPM package** (`.rpm`) from the download page.
2. Open a terminal and run:

   ```bash
   sudo rpm -ivh dbeaver-ce_<version>.x86_64.rpm
   ```

3. Launch DBeaver from your applications menu or run `dbeaver` in the terminal.

#### Option C — Any Linux (Snap)

```bash
sudo snap install dbeaver-ce
```

---

## Setting Up a Database Connection

When DBeaver launches for the first time, it will ask you to create a connection.

### Recommended: SQLite (easiest — no server needed)

SQLite is the simplest option. It stores your entire database in a single file on your computer — no installation of a database server is required.

1. Click **New Database Connection** (the plug icon in the top left).
2. Search for **SQLite** and select it, then click **Next**.
3. Under **Path**, click **Create** to create a new database file, or **Open** to use an existing one.
4. Click **Finish**.
5. Your SQLite database is now connected and ready to use.

---

### Other Database Options

If you already have a database server installed or prefer a different engine, you can connect to:

| Database     | Notes                                      |
|--------------|--------------------------------------------|
| PostgreSQL   | Popular open-source, great for learning SQL |
| MySQL/MariaDB | Widely used for web applications           |
| SQLite       | No server needed, file-based               |
| H2 (embedded) | Built into DBeaver, no setup required     |

For any of these, click **New Database Connection**, select the database type, and fill in the **host**, **port**, **username**, and **password** provided by your instructor.

---

## Quick Test

Once connected, verify everything is working:

1. Expand your connection in the left panel (Database Navigator).
2. Click on your database name.
3. Go to **SQL Editor → New SQL Script** .
4. If you get a notification an SQL driver is missing, accept the driver download.
5. Type the following and press `Ctrl+Enter` to run it:

   ```sql
   SELECT 'Hello, DBeaver!' AS message;
   ```

6. If you see the result in the panel below, you are ready to go!

---

## Troubleshooting

| Problem | Solution |
|---|---|
| DBeaver won't open on macOS | Go to System Settings → Privacy & Security → Open Anyway |
| "Driver not found" error | In the connection dialog, click **Download** to auto-install the driver |
| Connection refused | Make sure your database server is running before connecting |
| Blank/white screen on Linux | Try launching with `dbeaver -nosplash` in terminal |

---

## Need Help?

- Official documentation: https://dbeaver.com/docs/dbeaver/
- Community forum: https://github.com/dbeaver/dbeaver/discussions

---

*Guide prepared for SQL class — DBeaver Community Edition 25.x*
