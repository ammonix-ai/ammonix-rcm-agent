import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio
from playwright.async_api import async_playwright
OUT = str(FIG / "ui_universe.jpg")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page = await ctx.new_page()
        await page.goto("http://127.0.0.1:8001/universe.html", wait_until="networkidle")
        await page.wait_for_timeout(3000)
        for label in ("Knowledge only", "2D"):
            try:
                await page.get_by_text(label, exact=True).first.click(); await page.wait_for_timeout(1500)
            except Exception as e:
                print("could not click", label, e)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=OUT, full_page=True, type="jpeg", quality=92)
        print("saved")
        await b.close()
asyncio.run(main())
