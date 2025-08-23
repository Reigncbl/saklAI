import React from 'react';

/**
 * Reusable Admin Chat Card component.
 * Props:
 * - user: string (e.g. "User #1")
 * - inquiry: string (e.g. "Loan Info Inquiry")
 * - percent: number (0-100)
 * - time: string (e.g. "2 mins ago")
 * - status: "takeover" | "assigned" | "complex"
 * - customerMsg: string
 * - agentMsg: string
 * - onAction: function (callback for button click)
 */
export default function AdminChatCard({
    user,
    inquiry,
    percent,
    time,
    status,
    customerMsg,
    agentMsg,
    onAction,
}) {
    // Determine styles
    const isYellow = status === "assigned" || status === "complex";
    const percentClass =
        percent === 0 ? "percent zero" :
            isYellow ? "percent yellow" :
                percent < 100 ? "percent low" : "percent";
    return (
        <div className={`chat-card${isYellow ? ' yellow' : ''}`}>
            <div>
                <div className="chat-header">
                    <div>
                        <div className="chat-user">{user}</div>
                        <div className="chat-type">{inquiry}</div>
                    </div>
                    <div className="chat-meta">
                        <span className={percentClass}>{percent}%</span>
                        <span className="time">{time}</span>
                    </div>
                </div>
                <div className="chat-content">
                    <div className="chat-bubble">{customerMsg}</div>
                    {status === "complex" ? (
                        <div className="chat-bubble yellow">
                            Complex Case Detected. Assigning to human agent...
                        </div>
                    ) : (
                        <div className="chat-bubble agent">{agentMsg}</div>
                    )}
                </div>
            </div>
            <div className="chat-footer">
                {status === "takeover" && (
                    <button className="btn takeover" onClick={onAction}>
                        Take Over
                    </button>
                )}
                {(status === "assigned" || status === "complex") && (
                    <button className="btn assigned" disabled>
                        Assigned
                    </button>
                )}
            </div>
        </div>
    );
}

