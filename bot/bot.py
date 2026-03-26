import discord
import os
from discord.ext import commands
from discord import app_commands
from datetime import datetime
from dotenv import load_dotenv
from predictor import ToxicityPredictor

# Load secret token from .env file
load_dotenv()
TOKEN     = os.getenv("DISCORD_TOKEN")
LOG_CHAN  = os.getenv("LOG_CHANNEL_NAME", "mod-logs")
THRESHOLD = int(os.getenv("TOXICITY_THRESHOLD", "50"))

# Initialise bot with all required intents (permissions)
intents = discord.Intents.default()
intents.message_content = True   # needed to read message text
intents.members = True            # needed to DM users

bot = commands.Bot(command_prefix="!", intents=intents)
predictor = ToxicityPredictor()    # load the ML model once at startup

# Track warnings per user (in-memory; resets on restart)
warning_count = {}


@bot.event
async def on_ready():
    guild = discord.Object(id=1427250454693412936)
    await bot.tree.sync(guild=guild)
    print("Slash commands synced")
    print(f"Logged in as {bot.user}")
    print(f"✅ Bot online as: {bot.user}")
    print(f"   Toxicity threshold: {THRESHOLD}%")
    print(f"   Logging to channel: #{LOG_CHAN}")


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself (prevent infinite loops)
    if message.author == bot.user:
        return

    # Ignore very short messages (single chars, emojis, etc.)
    if len(message.content.strip()) < 3:
        return

    # ── Run ML prediction ────────────────────────────────────────────────
    result = predictor.predict(message.content)
    score  = result["score"]
    label  = result["label"]

    # ── Only act if message is above the toxicity threshold ───────────────
    if result["is_toxic"]:
        user_id = message.author.id

        # Track how many warnings this user has received
        warning_count[user_id] = warning_count.get(user_id, 0) + 1
        warns = warning_count[user_id]

        # Step 1: Delete the toxic message
        try:
            await message.delete()
            print(f"[DELETED] {message.author}: {message.content[:60]}")
        except discord.Forbidden:
            print("⚠ Bot lacks permission to delete messages")

        # Step 2: Warn the user in the channel
        warn_msg = (
            f"⚠️ **{message.author.mention}**, your message was flagged as toxic "
            f"(**{score}% confidence**). This is warning **#{warns}**.\n"
            f"Please keep our community respectful."
        )
        await message.channel.send(warn_msg, delete_after=10)
        # delete_after=10 removes the warning after 10 seconds to keep chat clean

        # Step 3: Send a DM to the user for context
        try:
            dm_text = (
                f"Hi {message.author.name}, your message in "
                f"**{message.guild.name}** was removed.\n\n"
                f"**Reason:** Toxic content detected ({score}% confidence)\n"
                f"**Your message:** {message.content[:200]}\n\n"
                f"Please review the server rules."
            )
            await message.author.send(dm_text)
        except discord.Forbidden:
            pass  # user may have DMs disabled

        # Step 4: Log to the mod-logs channel
        log_channel = discord.utils.get(message.guild.channels, name=LOG_CHAN)
        if log_channel:
            embed = discord.Embed(
                title="🚨 Toxic Message Detected",
                color=discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="User",    value=f"{message.author} (ID: {user_id})",   inline=False)
            embed.add_field(name="Channel", value=message.channel.mention,              inline=True)
            embed.add_field(name="Score",   value=f"{score}% toxic",                    inline=True)
            embed.add_field(name="Warnings",value=f"#{warns} total",                    inline=True)
            embed.add_field(name="Message", value=message.content[:1000] or "(empty)", inline=False)
            embed.set_footer(text="ToxicGuard ML Bot")
            await log_channel.send(embed=embed)

        # Escalation: if 3+ warnings, notify mods to consider a timeout/ban
        if warns >= 3:
            if log_channel:
                await log_channel.send(
                    f"🔴 @here **{message.author.mention}** has received "
                    f"**{warns} warnings**. Consider manual action."
                )

@bot.tree.command(name="toxicity", description="Check toxicity of a message")
async def toxicity(interaction: discord.Interaction, text: str):
    result = predictor.predict(text)

    score = result["score"]
    label = result["label"]

    await interaction.response.send_message(
        f"🧠 **Toxicity Check**\n"
        f"Message: {text}\n"
        f"Score: **{score}%**\n"
        f"Label: **{label.upper()}**"
    )

# Run the bot (this is the last line — it starts the event loop)
bot.run(TOKEN)
