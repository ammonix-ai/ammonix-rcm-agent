"""Retake ui_dialog.jpg: the operator dialog on ep-05072-s1, answered live by
the pinned Qwen 3.8 27B at :8000. Asks the six questions of the original
figure and stitches two clean columns (Q1-3 | Q4-6 with the input row)."""
import pathlib
FIG = pathlib.Path(__file__).resolve().parent / "figures"
import asyncio
from playwright.async_api import async_playwright

QUESTIONS = [
    "Why did you request a peer-to-peer review instead of appealing the denial?",
    "Only 14% of similar cases collected. Why act at all instead of writing this off?",
    "Which past cases did you look at, and did any of them succeed?",
    "Did any hard rule force this action, or was it the ranking?",
    "Should I bill the patient instead?",
    "What is the patient's date of birth?",
]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1400, "height": 2400},
                                  device_scale_factor=2)
        page = await ctx.new_page()
        page.set_default_timeout(300_000)
        await page.goto("http://127.0.0.1:8001/case.html?id=ep-05072-s1",
                        wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await page.click("#chat-head")
        await page.add_style_tag(content=(
            ".chat-widget{width:41rem}"
            ".chat-log{max-height:none}"
        ))
        widget = page.locator("#chat-widget")
        for i, q in enumerate(QUESTIONS):
            await page.fill("#chat-q", q)
            await page.click("#chat-form button[type=submit]")
            await page.wait_for_function(
                "() => !document.querySelector('#chat-log .pending')"
                " && !document.getElementById('chat-q').disabled",
                timeout=300_000)
            await page.wait_for_timeout(400)
            print("answered", i + 1)
            if i == 2:
                # column 1: Q1-3, no input row
                await page.evaluate(
                    "document.querySelector('#chat-form').style.display='none'")
                await page.wait_for_timeout(250)
                await widget.screenshot(path=str(FIG / "_dialog_col1.png"))
                await page.evaluate(
                    "document.querySelector('#chat-form').style.display=''")
        # column 2: Q4-6 and the input row (hide the first three exchanges)
        await page.evaluate(
            "() => { const kids = document.querySelectorAll('#chat-log .msg');"
            " for (let j = 0; j < 6; j++) kids[j].style.display = 'none'; }")
        await page.wait_for_timeout(250)
        await widget.screenshot(path=str(FIG / "_dialog_col2.png"))
        await b.close()

    from PIL import Image
    a = Image.open(FIG / "_dialog_col1.png")
    c = Image.open(FIG / "_dialog_col2.png")
    h = max(a.height, c.height)
    gap = 40
    out = Image.new("RGB", (a.width + c.width + gap, h), "white")
    out.paste(a, (0, 0))
    out.paste(c, (a.width + gap, 0))
    out.save(FIG / "ui_dialog.jpg", quality=92)
    print("saved ui_dialog.jpg", out.size)

asyncio.run(main())
