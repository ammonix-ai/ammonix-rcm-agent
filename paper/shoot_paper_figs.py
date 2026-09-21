"""Retake the paper's three page-level UI screenshots on the current demo
(GPT-6 lane, 2026-09 UI). 2x device scale, same as the earlier figures.
Server: uvicorn ui.server:app --port 8001 with AMMONIX_LLM_LANE=gpt6."""
import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8001"

async def hide_chat(page):
    try:
        await page.add_style_tag(content="#chat-widget{display:none !important}")
    except Exception:
        pass

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1600, "height": 1000},
                                  device_scale_factor=2)
        page = await ctx.new_page()

        # 1. per-claim audit view (paper caption case: ep-05091-s0)
        await page.goto(f"{BASE}/case.html?id=ep-05091-s0", wait_until="networkidle")
        await page.wait_for_timeout(3000)
        await hide_chat(page)
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(FIG / "ui_audit_trail.jpg"),
                              full_page=True, type="jpeg", quality=92)
        print("saved ui_audit_trail.jpg")

        # 2. three-lane replay, ep-05103
        await page.goto(f"{BASE}/claim.html?id=ep-05103", wait_until="networkidle")
        await page.wait_for_timeout(3000)
        await hide_chat(page)
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(FIG / "ui_replay.jpg"),
                              full_page=True, type="jpeg", quality=92)
        print("saved ui_replay.jpg")

        # 3. Knowledge Universe, training cases with cluster labels
        await page.goto(f"{BASE}/universe.html", wait_until="networkidle")
        await page.wait_for_timeout(4000)  # map + labels draw
        for label in ("Training cases",):
            try:
                await page.get_by_text(label, exact=True).first.click()
                await page.wait_for_timeout(2000)
            except Exception as e:
                print("could not click", label, e)
        await hide_chat(page)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(FIG / "ui_universe.jpg"),
                              full_page=True, type="jpeg", quality=92)
        print("saved ui_universe.jpg")

        await b.close()

asyncio.run(main())
