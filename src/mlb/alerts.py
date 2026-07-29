"""Alert notifications — Slack and email for value bets.

Sends notifications when the daily runner finds high-confidence value bets.

Configuration via environment variables:
    SLACK_WEBHOOK_URL  — Slack incoming webhook URL
    ALERT_EMAIL_TO     — Email recipient
    SMTP_HOST          — SMTP server (default: smtp.gmail.com)
    SMTP_PORT          — SMTP port (default: 587)
    SMTP_USER          — SMTP username
    SMTP_PASSWORD      — SMTP password (app password for Gmail)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AlertService:
    """Sends alerts via Slack and/or email."""

    def __init__(self):
        self.slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
        self.email_to = os.environ.get("ALERT_EMAIL_TO", "")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook)

    @property
    def email_enabled(self) -> bool:
        return bool(self.email_to and self.smtp_user and self.smtp_password)

    async def send_alert(self, text: str):
        """Send a generic text alert to Slack."""
        if self.slack_enabled:
            payload = {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                ],
            }
            await self._send_slack(payload)
        else:
            logger.info("Alert (no Slack): %s", text)

    async def send_betting_alert(self, result: dict[str, Any]):
        """Send alerts for today's predictions and value bets."""
        preds = result.get("predictions", [])
        slip = result.get("betting_slip")

        if not preds:
            logger.info("No predictions to alert on")
            return

        logger.info(
            "Sending alerts (slack=%s, email=%s, %d predictions)",
            self.slack_enabled, self.email_enabled, len(preds),
        )

        message = self._format_message(result)
        slack_blocks = self._format_slack_blocks(result)

        if self.slack_enabled:
            await self._send_slack(slack_blocks)
        else:
            logger.warning("Slack not enabled — SLACK_WEBHOOK_URL is empty")
        if self.email_enabled:
            self._send_email(
                subject=f"MLB Predictions — {result['date']}",
                body=message,
            )

    def _format_message(self, result: dict) -> str:
        """Format a plain-text summary."""
        lines = [self._card_title(result), "=" * 40, ""]

        review = result.get("review")
        if review:
            lines.append(f"Yesterday ({review.get('yesterday', '?')}) review:")
            lines.append(f"  Full slate:        {self._record_str(review.get('full'))}")
            lines.append(f"  High-conf (≥65):   {self._record_str(review.get('high_conf'))}")
            win = review.get("high_conf_window") or {}
            lines.append(f"  High-conf last {win.get('days', 5)}d: {self._record_str(win)}")
            card = review.get("card")
            if card:
                rec = f"{card['won']}W-{card['lost']}L" + (f"-{card['pushed']}P" if card.get("pushed") else "")
                lines.append(f"  Betting card:     {rec}  ${card['pnl']:+,.2f} ({card['roi']:+.1%})")
            else:
                lines.append("  Betting card:     no bets settled")
            lines.append("")

        for p in result.get("predictions", []):
            winner = p["predicted_winner"]
            prob = max(p["home_win_prob"], p["away_win_prob"])
            h_sp = p.get("home_sp_name", "TBD")
            a_sp = p.get("away_sp_name", "TBD")
            lines.append(
                f"{p['away_team']} @ {p['home_team']}  |  "
                f"{winner} {prob:.1%}  |  Conf: {p['confidence']:.0f}"
            )
            lines.append(f"  SP: {a_sp} vs {h_sp}")

        slip = result.get("betting_slip")
        if slip and slip.get("num_bets", 0) > 0:
            lines.append("")
            lines.append(f"VALUE BETS ({slip['num_bets']})")
            lines.append(f"Total Stake: ${slip['total_stake']:.2f}  |  EV: ${slip['total_ev']:.2f}")
            for b in slip["bets"]:
                lines.append(
                    f"  {b['bet_type'].upper()} {b['selection']}  "
                    f"{b['home_team']}v{b['away_team']}  "
                    f"Edge: {b['edge_pct']:.1f}%  Stake: ${b['recommended_stake']:.2f}"
                )

        return "\n".join(lines)

    @staticmethod
    def _record_str(rec: dict | None) -> str:
        """Render a {correct,incorrect,total} tally as 'W-L (pct)'."""
        if not rec or not rec.get("total"):
            return "no graded games"
        pct = rec["correct"] / rec["total"]
        return f"{rec['correct']}-{rec['incorrect']} ({pct:.1%})"

    def _format_review_text(self, review: dict) -> str:
        """Build the yesterday-review mrkdwn block for the top of the card."""
        lines = [f":bar_chart: *Yesterday ({review.get('yesterday', '?')}) — Review*"]
        # "Full slate", not "full card" — the betting card is the line below it, and
        # having both read as "card" is exactly what made these reports hard to read.
        lines.append(f"• Full slate: {self._record_str(review.get('full'))}")
        lines.append(f"• High-conf (≥65): {self._record_str(review.get('high_conf'))}")
        win = review.get("high_conf_window") or {}
        days = win.get("days", 5)
        lines.append(f"• High-conf, last {days}d: {self._record_str(win)}")

        card = review.get("card")
        if card:
            rec = f"{card['won']}W-{card['lost']}L"
            if card.get("pushed"):
                rec += f"-{card['pushed']}P"
            emoji = ":chart_with_upwards_trend:" if card["pnl"] >= 0 else ":chart_with_downwards_trend:"
            lines.append(
                f"• Betting card: {emoji} {rec} · "
                f"${card['pnl']:+,.2f} ({card['roi']:+.1%}) on ${card['staked']:,.2f}"
            )
        else:
            lines.append("• Betting card: no bets settled")
        return "\n".join(lines)

    @staticmethod
    def _card_title(result: dict) -> str:
        """Header text — depends on whether this is a morning preview or a wave send."""
        if result.get("kind") == "wave":
            return f"Starting Soon — {result['date']}"
        return f"MLB Predictions — {result['date']}"

    def _format_slack_blocks(self, result: dict) -> dict:
        """Format Slack Block Kit message."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": self._card_title(result)},
            }
        ]

        # Yesterday's review (record + betting card), if available
        review = result.get("review")
        if review:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": self._format_review_text(review)},
            })
            blocks.append({"type": "divider"})

        # Game predictions
        pred_lines = []
        for p in result.get("predictions", []):
            winner = p["predicted_winner"]
            prob = max(p["home_win_prob"], p["away_win_prob"])
            emoji = ":fire:" if p["confidence"] >= 60 else ":baseball:"
            h_sp = p.get("home_sp_name", "TBD")
            a_sp = p.get("away_sp_name", "TBD")
            pred_lines.append(
                f"{emoji} *{p['away_team']} @ {p['home_team']}* — "
                f"{winner} {prob:.1%} (Conf: {p['confidence']:.0f})\n"
                f"    _{a_sp} vs {h_sp}_"
            )

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(pred_lines)},
        })

        # Wave cards are near-first-pitch reminders — the whole day's card went out
        # on the main send, so say so rather than leaving a bet-less card ambiguous.
        if result.get("kind") == "wave":
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": ":pushpin: Reminder only — today's bets went out on the "
                            "main card.",
                }],
            })

        # Value bets
        slip = result.get("betting_slip")
        if slip and slip.get("num_bets", 0) > 0:
            blocks.append({"type": "divider"})
            bet_lines = [
                f":moneybag: *{slip['num_bets']} Value Bets* — "
                f"Stake: ${slip['total_stake']:.2f} | EV: ${slip['total_ev']:.2f}"
            ]
            for b in slip["bets"]:
                bet_lines.append(
                    f"  :point_right: {b['bet_type'].upper()} {b['selection']}  "
                    f"{b['home_team']}v{b['away_team']}  "
                    f"Edge: {b['edge_pct']:.1f}%  ${b['recommended_stake']:.2f}"
                )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(bet_lines)},
            })

        return {"blocks": blocks}

    async def send_settlement_alert(self, settlement: dict):
        """Send alerts for settlement results."""
        summary = settlement.get("summary", {})
        if not summary.get("bets_placed"):
            return

        message = self._format_settlement_message(settlement)
        slack_blocks = self._format_settlement_slack_blocks(settlement)

        if self.slack_enabled:
            await self._send_slack(slack_blocks)
        if self.email_enabled:
            self._send_email(
                subject=f"MLB Settlement — {settlement['date']}",
                body=message,
            )

    def _format_settlement_message(self, settlement: dict) -> str:
        """Format plain-text settlement summary."""
        s = settlement["summary"]
        lines = [
            f"MLB Settlement for {settlement['date']}",
            "=" * 40,
            "",
            f"Record: {s['bets_won']}W-{s['bets_lost']}L-{s['bets_pushed']}P",
            f"Staked: ${s['total_staked']:.2f}",
            f"Daily P&L: ${s['daily_pnl']:+.2f}",
            f"ROI: {s['roi']:.1%}",
            f"Cumulative P&L: ${s['cumulative_pnl']:+.2f}",
            f"Max Drawdown: {s['max_drawdown']:.1%}",
            "",
        ]
        for b in settlement.get("bets", []):
            tag = "W" if b["result"] == "win" else ("L" if b["result"] == "loss" else "P")
            lines.append(
                f"  [{tag}] {b['bet_type'].upper()} {b['selection']}  "
                f"{b['away_team']}@{b['home_team']}  "
                f"{b['actual_away_score']}-{b['actual_home_score']}  "
                f"${b['pnl']:+.2f}"
            )
        return "\n".join(lines)

    def _format_settlement_slack_blocks(self, settlement: dict) -> dict:
        """Format Slack Block Kit settlement message."""
        s = settlement["summary"]
        pnl_emoji = ":chart_with_upwards_trend:" if s["daily_pnl"] >= 0 else ":chart_with_downwards_trend:"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Settlement Results — {settlement['date']}"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{pnl_emoji} *{s['bets_won']}W-{s['bets_lost']}L-{s['bets_pushed']}P*\n"
                        f"Daily P&L: *${s['daily_pnl']:+.2f}* ({s['roi']:.1%})\n"
                        f"Cumulative: *${s['cumulative_pnl']:+.2f}*"
                    ),
                },
            },
            {"type": "divider"},
        ]

        bet_lines = []
        for b in settlement.get("bets", []):
            if b["result"] == "win":
                emoji = ":white_check_mark:"
            elif b["result"] == "loss":
                emoji = ":x:"
            else:
                emoji = ":heavy_minus_sign:"
            bet_lines.append(
                f"{emoji} {b['bet_type'].upper()} {b['selection']}  "
                f"*{b['away_team']}@{b['home_team']}*  "
                f"{b['actual_away_score']}-{b['actual_home_score']}  "
                f"${b['pnl']:+.2f}"
            )

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(bet_lines)},
        })

        return {"blocks": blocks}

    async def _send_slack(self, payload: dict):
        """Send a Slack webhook message."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.slack_webhook, json=payload)
                if resp.status_code == 200:
                    logger.info("Slack alert sent")
                else:
                    logger.warning("Slack alert failed: %s %s", resp.status_code, resp.text)
        except Exception:
            logger.exception("Slack alert error")

    def _send_email(self, subject: str, body: str):
        """Send an email alert via SMTP."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            msg.attach(MIMEText(body, "plain"))

            # Simple HTML version
            html_body = f"<pre style='font-family:monospace'>{body}</pre>"
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info("Email alert sent to %s", self.email_to)
        except Exception:
            logger.exception("Email alert error")
