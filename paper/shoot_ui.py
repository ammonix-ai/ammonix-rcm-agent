"""Full-page screenshots of the v0.6 demo UI at 2x device scale (matches the
Aug-9 figures' 3200px width). Collapses the Ask Ammonix chat overlay first."""
import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio, sys
from playwright.async_api import async_playwright

SHOTS = {
    "ui_audit_trail.jpg": "http://127.0.0.1:8001/case.html?id=ep-05095-r0",
    "ui_replay.jpg":      "http://127.0.0.1:8001/claim.html?id=ep-05103",
    "ui_universe.jpg":    "http://127.0.0.1:8001/universe.html",
}
OUT = str(FIG)

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page = await ctx.new_page()
        for name, url in SHOTS.items():
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(2500)
            # collapse the chat overlay if present (button text "–" in the Ask Ammonix header)
            try:
                btn = page.locator("text=Ask Ammonix").locator("xpath=..").locator("button").first
                if await btn.count() > 0:
                    await btn.click(); await page.wait_for_timeout(300)
            except Exception:
                pass
            await page.screenshot(path=f"{OUT}\\{name}", full_page=True, type="jpeg", quality=92)
            print("saved", name)
        await b.close()

asyncio.run(main())
