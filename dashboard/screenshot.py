"""
Take screenshots of all three Streamlit dashboard pages.
Run after `streamlit run app.py` is already serving on localhost:8501.
"""
import asyncio
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

async def screenshot_all():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            color_scheme="dark",
        )
        page = await ctx.new_page()

        pages = [
            ("Circular+Financing+Map", "screenshot_financing_map.png"),
            ("P%26L+Monte+Carlo",      "screenshot_monte_carlo.png"),
            ("Earnings+Sentiment",     "screenshot_earnings_sentiment.png"),
        ]

        for nav_label, filename in pages:
            url = f"http://localhost:8501/?page={nav_label}"
            print(f"  Navigating to {url} ...", flush=True)
            await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)

            # Click the correct radio button in sidebar
            label_map = {
                "Circular+Financing+Map": "Circular Financing Map",
                "P%26L+Monte+Carlo":      "P&L Monte Carlo",
                "Earnings+Sentiment":     "Earnings Sentiment",
            }
            label_text = label_map[nav_label]
            try:
                await page.get_by_text(label_text, exact=True).click(timeout=8000)
                # Wait for spinner to disappear (Streamlit finishes running)
                try:
                    await page.wait_for_selector('[data-testid="stSpinner"]', timeout=3000)
                    await page.wait_for_selector('[data-testid="stSpinner"]',
                                                state="hidden", timeout=20000)
                except Exception:
                    pass
                await page.wait_for_timeout(4500)   # final settle
            except Exception as e:
                print(f"  [!] Could not click nav item '{label_text}': {e}")

            out_path = str(OUTPUT_DIR / filename)
            await page.screenshot(path=out_path, full_page=False)
            print(f"  Saved -> {out_path}")

        await browser.close()
        print("\nAll screenshots saved.")

if __name__ == "__main__":
    asyncio.run(screenshot_all())
