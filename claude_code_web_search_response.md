**Current release: Python 3.15.** The latest patch is 3.15.2, released 25 Oct 2026. The official docs at docs.python.org/3 are now the 3.15 documentation. 3.15.0 came out on 9 Oct 2026.

The two searches gave slightly different support dates for 3.15, so check the devguide before you rely on them. One result put end of life at October 2031 with bugfixes until October 2028. Another said bugfix support runs until 2031-10. I didn't open the devguide page myself.

**Why a team might choose it for a new service:**
- **Longest support runway.** A service started now has security support into 2031, so there is no forced upgrade for years.
- **Newest language and standard-library features,** plus ongoing runtime improvements.
- **No legacy baggage.** A new service has no old code to migrate, so the usual cost of adopting a new version is small.

**Caveats:** 3.15 is only weeks old. Check that your key dependencies, especially compiled ones, publish 3.15 wheels. If they don't, 3.14 is a safer choice. It is still in bugfix support and is supported until October 2030.

Sources:
- [Status of Python versions (devguide)](https://devguide.python.org/versions/)
- [Python 3.15.2 documentation](https://docs.python.org/3/download.html)
- [Python Release 3.15.0](https://www.python.org/downloads/release/python-3150/)
- [Python Release 3.14.10](https://www.python.org/downloads/latest/python3.14/)
- [Download Python](https://www.python.org/downloads/)
