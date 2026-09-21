"""Universe retake: training cases, zoomed so the galaxy fills the stage."""
import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1600, "height": 1000},
                                  device_scale_factor=2)
        page = await ctx.new_page()
        await page.goto("http://127.0.0.1:8001/universe.html",
                        wait_until="networkidle")
        await page.wait_for_timeout(4000)
        try:
            await page.get_by_text("Training cases", exact=True).first.click()
            await page.wait_for_timeout(2000)
        except Exception as e:
            print("could not click Training cases", e)
        for _ in range(1):
            try:
                await page.get_by_text("+", exact=True).first.click()
                await page.wait_for_timeout(800)
            except Exception as e:
                print("could not click +", e)
                break
        try:
            await page.add_style_tag(content="#chat-widget{display:none !important}")
        except Exception:
            pass
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(FIG / "ui_universe.jpg"),
                              full_page=True, type="jpeg", quality=92)
        print("saved")
        await b.close()

asyncio.run(main())
