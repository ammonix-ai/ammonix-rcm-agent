import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio
from playwright.async_api import async_playwright
OUT = str(FIG / "ui_audit_trail.jpg")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page = await ctx.new_page()
        await page.goto("http://127.0.0.1:8001/case.html?id=ep-05095-r0", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await page.add_style_tag(content="#chat-widget{display:none !important}")
        await page.wait_for_timeout(300)
        await page.screenshot(path=OUT, full_page=True, type="jpeg", quality=92)
        print("saved")
        await b.close()
asyncio.run(main())
